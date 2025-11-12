package monitoring

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// DashboardExporter exports metrics in Grafana-compatible format
type DashboardExporter struct {
	mu sync.RWMutex

	metricsCollector *DWCPv3MetricsCollector
	perfTracker      *PerformanceTracker
	anomalyDetector  *AnomalyDetector

	logger *zap.Logger
}

// GrafanaDashboard represents a complete Grafana dashboard configuration
type GrafanaDashboard struct {
	UID           string           `json:"uid"`
	Title         string           `json:"title"`
	Tags          []string         `json:"tags"`
	Timezone      string           `json:"timezone"`
	SchemaVersion int              `json:"schemaVersion"`
	Version       int              `json:"version"`
	Refresh       string           `json:"refresh"`
	Panels        []GrafanaPanel   `json:"panels"`
	Time          GrafanaTimeRange `json:"time"`
}

// GrafanaPanel represents a Grafana dashboard panel
type GrafanaPanel struct {
	ID          int                    `json:"id"`
	Title       string                 `json:"title"`
	Type        string                 `json:"type"` // graph, singlestat, table, heatmap
	GridPos     GrafanaGridPos         `json:"gridPos"`
	Targets     []GrafanaTarget        `json:"targets"`
	Options     map[string]interface{} `json:"options,omitempty"`
	FieldConfig map[string]interface{} `json:"fieldConfig,omitempty"`
}

// GrafanaGridPos defines panel position and size
type GrafanaGridPos struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"` // Width (1-24)
	H int `json:"h"` // Height
}

// GrafanaTarget defines a metric query
type GrafanaTarget struct {
	Expr           string `json:"expr"`
	LegendFormat   string `json:"legendFormat,omitempty"`
	RefID          string `json:"refId"`
	Interval       string `json:"interval,omitempty"`
}

// GrafanaTimeRange defines dashboard time range
type GrafanaTimeRange struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// NewDashboardExporter creates a new dashboard exporter
func NewDashboardExporter(
	metricsCollector *DWCPv3MetricsCollector,
	perfTracker *PerformanceTracker,
	anomalyDetector *AnomalyDetector,
	logger *zap.Logger,
) *DashboardExporter {
	return &DashboardExporter{
		metricsCollector: metricsCollector,
		perfTracker:      perfTracker,
		anomalyDetector:  anomalyDetector,
		logger:           logger,
	}
}

// ExportMainDashboard exports the main DWCP v3 monitoring dashboard
func (de *DashboardExporter) ExportMainDashboard() ([]byte, error) {
	dashboard := GrafanaDashboard{
		UID:           "dwcp-v3-main",
		Title:         "DWCP v3 Monitoring Dashboard",
		Tags:          []string{"dwcp", "v3", "migration", "performance"},
		Timezone:      "browser",
		SchemaVersion: 16,
		Version:       1,
		Refresh:       "10s",
		Time: GrafanaTimeRange{
			From: "now-1h",
			To:   "now",
		},
		Panels: []GrafanaPanel{
			de.createOverviewPanel(),
			de.createModeDistributionPanel(),
			de.createThroughputPanel(),
			de.createLatencyPanel(),
			de.createComponentHealthPanel(),
			de.createCompressionPanel(),
			de.createAnomaliesPanel(),
			de.createV1vsV3ComparisonPanel(),
		},
	}

	return json.MarshalIndent(dashboard, "", "  ")
}

// ExportModeDashboard exports mode-specific dashboard
func (de *DashboardExporter) ExportModeDashboard(mode string) ([]byte, error) {
	dashboard := GrafanaDashboard{
		UID:           fmt.Sprintf("dwcp-v3-%s", mode),
		Title:         fmt.Sprintf("DWCP v3 %s Mode Dashboard", mode),
		Tags:          []string{"dwcp", "v3", mode},
		Timezone:      "browser",
		SchemaVersion: 16,
		Version:       1,
		Refresh:       "5s",
		Time: GrafanaTimeRange{
			From: "now-30m",
			To:   "now",
		},
		Panels: de.getModePanels(mode),
	}

	return json.MarshalIndent(dashboard, "", "  ")
}

// ExportComponentDashboard exports component-specific dashboard
func (de *DashboardExporter) ExportComponentDashboard(component string) ([]byte, error) {
	dashboard := GrafanaDashboard{
		UID:           fmt.Sprintf("dwcp-v3-%s", component),
		Title:         fmt.Sprintf("DWCP v3 %s Component Dashboard", component),
		Tags:          []string{"dwcp", "v3", component},
		Timezone:      "browser",
		SchemaVersion: 16,
		Version:       1,
		Refresh:       "5s",
		Time: GrafanaTimeRange{
			From: "now-30m",
			To:   "now",
		},
		Panels: de.getComponentPanels(component),
	}

	return json.MarshalIndent(dashboard, "", "  ")
}

// Panel creation methods

func (de *DashboardExporter) createOverviewPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    1,
		Title: "DWCP v3 Overview",
		Type:  "stat",
		GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 4},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_migration_success_total",
				LegendFormat: "Successful Migrations",
				RefID:        "A",
			},
			{
				Expr:         "rate(dwcp_v3_bytes_transferred_total[5m])",
				LegendFormat: "Throughput (Mbps)",
				RefID:        "B",
			},
			{
				Expr:         "dwcp_v3_active_streams",
				LegendFormat: "Active Streams",
				RefID:        "C",
			},
		},
	}
}

func (de *DashboardExporter) createModeDistributionPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    2,
		Title: "Network Mode Distribution",
		Type:  "piechart",
		GridPos: GrafanaGridPos{X: 0, Y: 4, W: 8, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "sum by (mode) (dwcp_v3_migration_success_total)",
				LegendFormat: "{{mode}}",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createThroughputPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    3,
		Title: "Throughput by Mode",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 8, Y: 4, W: 16, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "rate(dwcp_v3_bytes_transferred_total{mode=\"datacenter\"}[1m]) * 8 / 1000000",
				LegendFormat: "Datacenter",
				RefID:        "A",
			},
			{
				Expr:         "rate(dwcp_v3_bytes_transferred_total{mode=\"internet\"}[1m]) * 8 / 1000000",
				LegendFormat: "Internet",
				RefID:        "B",
			},
			{
				Expr:         "rate(dwcp_v3_bytes_transferred_total{mode=\"hybrid\"}[1m]) * 8 / 1000000",
				LegendFormat: "Hybrid",
				RefID:        "C",
			},
		},
	}
}

func (de *DashboardExporter) createLatencyPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    4,
		Title: "Latency Distribution (P50, P95, P99)",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 0, Y: 12, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "histogram_quantile(0.50, dwcp_v3_mode_latency_seconds_bucket)",
				LegendFormat: "P50",
				RefID:        "A",
			},
			{
				Expr:         "histogram_quantile(0.95, dwcp_v3_mode_latency_seconds_bucket)",
				LegendFormat: "P95",
				RefID:        "B",
			},
			{
				Expr:         "histogram_quantile(0.99, dwcp_v3_mode_latency_seconds_bucket)",
				LegendFormat: "P99",
				RefID:        "C",
			},
		},
	}
}

func (de *DashboardExporter) createComponentHealthPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    5,
		Title: "Component Health Status",
		Type:  "table",
		GridPos: GrafanaGridPos{X: 12, Y: 12, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_component_operations_total",
				LegendFormat: "{{component}} - {{operation_type}}",
				RefID:        "A",
			},
			{
				Expr:         "dwcp_v3_component_errors_total",
				LegendFormat: "{{component}} Errors",
				RefID:        "B",
			},
		},
	}
}

func (de *DashboardExporter) createCompressionPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    6,
		Title: "Compression Ratio by Algorithm",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 0, Y: 20, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_compression_ratio",
				LegendFormat: "{{algorithm}}",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createAnomaliesPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    7,
		Title: "Detected Anomalies (Last 1h)",
		Type:  "table",
		GridPos: GrafanaGridPos{X: 12, Y: 20, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "sum by (component, metric) (increase(dwcp_v3_anomalies_total[1h]))",
				LegendFormat: "{{component}} - {{metric}}",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createV1vsV3ComparisonPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    8,
		Title: "v1 vs v3 Performance Comparison",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 0, Y: 28, W: 24, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "rate(dwcp_v1_bytes_transferred_total[1m]) * 8 / 1000000",
				LegendFormat: "v1 Throughput",
				RefID:        "A",
			},
			{
				Expr:         "rate(dwcp_v3_bytes_transferred_total[1m]) * 8 / 1000000",
				LegendFormat: "v3 Throughput",
				RefID:        "B",
			},
		},
	}
}

// Mode-specific panels

func (de *DashboardExporter) getModePanels(mode string) []GrafanaPanel {
	switch mode {
	case "datacenter":
		return []GrafanaPanel{
			de.createDatacenterRDMAPanel(),
			de.createDatacenterStreamsPanel(),
			de.createDatacenterLatencyHeatmap(),
		}
	case "internet":
		return []GrafanaPanel{
			de.createInternetTCPPanel(),
			de.createInternetCompressionPanel(),
			de.createInternetByzantinePanel(),
		}
	case "hybrid":
		return []GrafanaPanel{
			de.createHybridModeSwitchPanel(),
			de.createHybridAdaptiveMetricsPanel(),
			de.createHybridFailoverPanel(),
		}
	default:
		return []GrafanaPanel{}
	}
}

func (de *DashboardExporter) createDatacenterRDMAPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    10,
		Title: "RDMA Throughput and Bandwidth",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 0, Y: 0, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_rdma_throughput_gbps{mode=\"datacenter\"}",
				LegendFormat: "RDMA Throughput",
				RefID:        "A",
			},
			{
				Expr:         "dwcp_v3_rdma_bandwidth_utilization{mode=\"datacenter\"}",
				LegendFormat: "Bandwidth Utilization %",
				RefID:        "B",
			},
		},
	}
}

func (de *DashboardExporter) createDatacenterStreamsPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    11,
		Title: "Active RDMA Streams",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 12, Y: 0, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_active_streams{mode=\"datacenter\"}",
				LegendFormat: "Active Streams",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createDatacenterLatencyHeatmap() GrafanaPanel {
	return GrafanaPanel{
		ID:    12,
		Title: "Latency Heatmap (Datacenter)",
		Type:  "heatmap",
		GridPos: GrafanaGridPos{X: 0, Y: 8, W: 24, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_mode_latency_seconds_bucket{mode=\"datacenter\"}",
				LegendFormat: "{{le}}",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createInternetTCPPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    20,
		Title: "TCP Streams and Congestion",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 0, Y: 0, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_active_streams{mode=\"internet\"}",
				LegendFormat: "TCP Streams",
				RefID:        "A",
			},
			{
				Expr:         "rate(dwcp_v3_congestion_events_total{mode=\"internet\"}[5m])",
				LegendFormat: "Congestion Events/s",
				RefID:        "B",
			},
		},
	}
}

func (de *DashboardExporter) createInternetCompressionPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    21,
		Title: "Compression Ratio (Internet Mode)",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 12, Y: 0, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_compression_ratio{mode=\"internet\"}",
				LegendFormat: "Compression Ratio",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createInternetByzantinePanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    22,
		Title: "Byzantine Fault Detection",
		Type:  "stat",
		GridPos: GrafanaGridPos{X: 0, Y: 8, W: 24, H: 4},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_byzantine_detections_total{mode=\"internet\"}",
				LegendFormat: "Byzantine Faults",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createHybridModeSwitchPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    30,
		Title: "Mode Switch Frequency",
		Type:  "graph",
		GridPos: GrafanaGridPos{X: 0, Y: 0, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "rate(dwcp_v3_mode_switches_total[5m])",
				LegendFormat: "{{from_mode}} -> {{to_mode}}",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createHybridAdaptiveMetricsPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    31,
		Title: "Adaptive Decision Accuracy",
		Type:  "gauge",
		GridPos: GrafanaGridPos{X: 12, Y: 0, W: 12, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "dwcp_v3_prediction_accuracy{mode=\"hybrid\"}",
				LegendFormat: "Prediction Accuracy %",
				RefID:        "A",
			},
		},
	}
}

func (de *DashboardExporter) createHybridFailoverPanel() GrafanaPanel {
	return GrafanaPanel{
		ID:    32,
		Title: "Failover Events",
		Type:  "table",
		GridPos: GrafanaGridPos{X: 0, Y: 8, W: 24, H: 8},
		Targets: []GrafanaTarget{
			{
				Expr:         "increase(dwcp_v3_failover_events_total[1h])",
				LegendFormat: "{{component}}",
				RefID:        "A",
			},
		},
	}
}

// Component-specific panels

func (de *DashboardExporter) getComponentPanels(component string) []GrafanaPanel {
	switch component {
	case "amst":
		return de.getAMSTPanels()
	case "hde":
		return de.getHDEPanels()
	case "pba":
		return de.getPBAPanels()
	case "ass":
		return de.getASSPanels()
	case "acp":
		return de.getACPPanels()
	case "itp":
		return de.getITPPanels()
	default:
		return []GrafanaPanel{}
	}
}

func (de *DashboardExporter) getAMSTPanels() []GrafanaPanel {
	return []GrafanaPanel{
		{
			ID:    40,
			Title: "AMST Stream Management",
			Type:  "graph",
			GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 8},
			Targets: []GrafanaTarget{
				{Expr: "dwcp_v3_active_streams", LegendFormat: "Active", RefID: "A"},
				{Expr: "dwcp_v3_failed_streams_total", LegendFormat: "Failed", RefID: "B"},
			},
		},
	}
}

func (de *DashboardExporter) getHDEPanels() []GrafanaPanel {
	return []GrafanaPanel{
		{
			ID:    50,
			Title: "HDE Compression Performance",
			Type:  "graph",
			GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 8},
			Targets: []GrafanaTarget{
				{Expr: "dwcp_v3_compression_ratio", LegendFormat: "Ratio", RefID: "A"},
				{Expr: "rate(dwcp_v3_compressions_total[1m])", LegendFormat: "Rate", RefID: "B"},
			},
		},
	}
}

func (de *DashboardExporter) getPBAPanels() []GrafanaPanel {
	return []GrafanaPanel{
		{
			ID:    60,
			Title: "PBA Prediction Accuracy",
			Type:  "graph",
			GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 8},
			Targets: []GrafanaTarget{
				{Expr: "dwcp_v3_prediction_accuracy", LegendFormat: "Accuracy %", RefID: "A"},
			},
		},
	}
}

func (de *DashboardExporter) getASSPanels() []GrafanaPanel {
	return []GrafanaPanel{
		{
			ID:    70,
			Title: "ASS Synchronization Status",
			Type:  "graph",
			GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 8},
			Targets: []GrafanaTarget{
				{Expr: "rate(dwcp_v3_sync_operations_total[1m])", LegendFormat: "Sync Rate", RefID: "A"},
			},
		},
	}
}

func (de *DashboardExporter) getACPPanels() []GrafanaPanel {
	return []GrafanaPanel{
		{
			ID:    80,
			Title: "ACP Consensus Performance",
			Type:  "graph",
			GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 8},
			Targets: []GrafanaTarget{
				{Expr: "dwcp_v3_consensus_latency_seconds", LegendFormat: "Latency", RefID: "A"},
			},
		},
	}
}

func (de *DashboardExporter) getITPPanels() []GrafanaPanel {
	return []GrafanaPanel{
		{
			ID:    90,
			Title: "ITP Placement Quality",
			Type:  "gauge",
			GridPos: GrafanaGridPos{X: 0, Y: 0, W: 24, H: 8},
			Targets: []GrafanaTarget{
				{Expr: "dwcp_v3_placement_score", LegendFormat: "Score", RefID: "A"},
			},
		},
	}
}

// ExportPrometheusConfig exports Prometheus scrape configuration
func (de *DashboardExporter) ExportPrometheusConfig() ([]byte, error) {
	config := map[string]interface{}{
		"scrape_configs": []map[string]interface{}{
			{
				"job_name":        "dwcp-v3",
				"scrape_interval": "10s",
				"static_configs": []map[string]interface{}{
					{
						"targets": []string{"localhost:9090"},
						"labels": map[string]string{
							"environment": "production",
							"component":   "dwcp-v3",
						},
					},
				},
			},
		},
	}

	return json.MarshalIndent(config, "", "  ")
}

// GetDashboardList returns list of available dashboards
func (de *DashboardExporter) GetDashboardList() []map[string]string {
	return []map[string]string{
		{"uid": "dwcp-v3-main", "title": "DWCP v3 Main Dashboard"},
		{"uid": "dwcp-v3-datacenter", "title": "Datacenter Mode Dashboard"},
		{"uid": "dwcp-v3-internet", "title": "Internet Mode Dashboard"},
		{"uid": "dwcp-v3-hybrid", "title": "Hybrid Mode Dashboard"},
		{"uid": "dwcp-v3-amst", "title": "AMST Component Dashboard"},
		{"uid": "dwcp-v3-hde", "title": "HDE Component Dashboard"},
		{"uid": "dwcp-v3-pba", "title": "PBA Component Dashboard"},
		{"uid": "dwcp-v3-ass", "title": "ASS Component Dashboard"},
		{"uid": "dwcp-v3-acp", "title": "ACP Component Dashboard"},
		{"uid": "dwcp-v3-itp", "title": "ITP Component Dashboard"},
	}
}
