package monitoring

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
)

// DashboardConfig configures Grafana integration
type DashboardConfig struct {
	GrafanaURL  string
	APIKey      string
	OrgID       int
	FolderID    int
}

// DashboardManager manages Grafana dashboards
type DashboardManager struct {
	mu     sync.RWMutex
	config *DashboardConfig
	client *http.Client

	dashboards map[string]*Dashboard
}

// Dashboard represents a Grafana dashboard
type Dashboard struct {
	UID         string
	Title       string
	Description string
	Panels      []*Panel
	Variables   []*Variable
	Tags        []string
}

// Panel represents a dashboard panel
type Panel struct {
	ID          int
	Title       string
	Type        string // graph, stat, table, heatmap
	Datasource  string
	Targets     []*Target
	GridPos     GridPos
}

// Target represents a query target
type Target struct {
	RefID      string
	Expr       string // PromQL expression
	LegendFormat string
}

// GridPos defines panel position
type GridPos struct {
	X int
	Y int
	W int
	H int
}

// Variable represents a dashboard variable
type Variable struct {
	Name  string
	Type  string
	Query string
	Label string
}

// NewDashboardManager creates a new dashboard manager
func NewDashboardManager(config *DashboardConfig) *DashboardManager {
	return &DashboardManager{
		config:     config,
		client:     &http.Client{},
		dashboards: make(map[string]*Dashboard),
	}
}

// CreateGlobalOverviewDashboard creates the global overview dashboard
func (dm *DashboardManager) CreateGlobalOverviewDashboard() error {
	dashboard := &Dashboard{
		UID:         "dwcp-global-overview",
		Title:       "DWCP Global Overview",
		Description: "Global view of all regions",
		Tags:        []string{"dwcp", "global"},
	}

	// Add panels
	dashboard.Panels = []*Panel{
		{
			ID:    1,
			Title: "Total Requests (All Regions)",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(rate(dwcp_requests_total[5m])) by (region)",
					LegendFormat: "{{region}}",
				},
			},
			GridPos: GridPos{X: 0, Y: 0, W: 12, H: 8},
		},
		{
			ID:    2,
			Title: "Error Rate (All Regions)",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(rate(dwcp_errors_total[5m])) by (region) / sum(rate(dwcp_requests_total[5m])) by (region)",
					LegendFormat: "{{region}}",
				},
			},
			GridPos: GridPos{X: 12, Y: 0, W: 12, H: 8},
		},
		{
			ID:    3,
			Title: "P95 Latency (All Regions)",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "histogram_quantile(0.95, sum(rate(dwcp_latency_bucket[5m])) by (region, le))",
					LegendFormat: "{{region}}",
				},
			},
			GridPos: GridPos{X: 0, Y: 8, W: 12, H: 8},
		},
		{
			ID:    4,
			Title: "Active Connections",
			Type:  "stat",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(dwcp_active_connections) by (region)",
					LegendFormat: "{{region}}",
				},
			},
			GridPos: GridPos{X: 12, Y: 8, W: 12, H: 8},
		},
	}

	// Add variables
	dashboard.Variables = []*Variable{
		{
			Name:  "region",
			Type:  "query",
			Query: "label_values(dwcp_requests_total, region)",
			Label: "Region",
		},
	}

	dm.mu.Lock()
	dm.dashboards[dashboard.UID] = dashboard
	dm.mu.Unlock()

	return dm.uploadDashboard(dashboard)
}

// CreateRegionalDashboard creates a region-specific dashboard
func (dm *DashboardManager) CreateRegionalDashboard(region string) error {
	dashboard := &Dashboard{
		UID:         fmt.Sprintf("dwcp-region-%s", region),
		Title:       fmt.Sprintf("DWCP Region: %s", region),
		Description: fmt.Sprintf("Detailed metrics for %s region", region),
		Tags:        []string{"dwcp", "region", region},
	}

	dashboard.Panels = []*Panel{
		{
			ID:    1,
			Title: "Request Rate",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  fmt.Sprintf(`rate(dwcp_requests_total{region="%s"}[5m])`, region),
					LegendFormat: "{{operation}}",
				},
			},
			GridPos: GridPos{X: 0, Y: 0, W: 12, H: 8},
		},
		{
			ID:    2,
			Title: "Latency Distribution",
			Type:  "heatmap",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  fmt.Sprintf(`sum(increase(dwcp_latency_bucket{region="%s"}[1m])) by (le)`, region),
				},
			},
			GridPos: GridPos{X: 12, Y: 0, W: 12, H: 8},
		},
	}

	dm.mu.Lock()
	dm.dashboards[dashboard.UID] = dashboard
	dm.mu.Unlock()

	return dm.uploadDashboard(dashboard)
}

// CreateDWCPProtocolDashboard creates DWCP protocol metrics dashboard
func (dm *DashboardManager) CreateDWCPProtocolDashboard() error {
	dashboard := &Dashboard{
		UID:         "dwcp-protocol-metrics",
		Title:       "DWCP Protocol Metrics",
		Description: "Detailed DWCP protocol performance",
		Tags:        []string{"dwcp", "protocol"},
	}

	dashboard.Panels = []*Panel{
		{
			ID:    1,
			Title: "Message Types Distribution",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(rate(dwcp_messages_total[5m])) by (message_type)",
					LegendFormat: "{{message_type}}",
				},
			},
			GridPos: GridPos{X: 0, Y: 0, W: 12, H: 8},
		},
		{
			ID:    2,
			Title: "Compression Ratio",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "avg(dwcp_compression_ratio) by (region)",
					LegendFormat: "{{region}}",
				},
			},
			GridPos: GridPos{X: 12, Y: 0, W: 12, H: 8},
		},
		{
			ID:    3,
			Title: "Bandwidth Utilization",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(rate(dwcp_bytes_sent[5m])) by (region)",
					LegendFormat: "Sent - {{region}}",
				},
				{
					RefID: "B",
					Expr:  "sum(rate(dwcp_bytes_received[5m])) by (region)",
					LegendFormat: "Received - {{region}}",
				},
			},
			GridPos: GridPos{X: 0, Y: 8, W: 24, H: 8},
		},
	}

	dm.mu.Lock()
	dm.dashboards[dashboard.UID] = dashboard
	dm.mu.Unlock()

	return dm.uploadDashboard(dashboard)
}

// CreateLoadBalancerDashboard creates load balancer performance dashboard
func (dm *DashboardManager) CreateLoadBalancerDashboard() error {
	dashboard := &Dashboard{
		UID:         "dwcp-load-balancer",
		Title:       "DWCP Load Balancer Performance",
		Description: "Load balancer metrics and performance",
		Tags:        []string{"dwcp", "load-balancer"},
	}

	dashboard.Panels = []*Panel{
		{
			ID:    1,
			Title: "Backend Health",
			Type:  "stat",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(dwcp_backend_healthy) by (backend)",
					LegendFormat: "{{backend}}",
				},
			},
			GridPos: GridPos{X: 0, Y: 0, W: 12, H: 8},
		},
		{
			ID:    2,
			Title: "Request Distribution",
			Type:  "graph",
			Datasource: "Prometheus",
			Targets: []*Target{
				{
					RefID: "A",
					Expr:  "sum(rate(dwcp_lb_requests_total[5m])) by (backend)",
					LegendFormat: "{{backend}}",
				},
			},
			GridPos: GridPos{X: 12, Y: 0, W: 12, H: 8},
		},
	}

	dm.mu.Lock()
	dm.dashboards[dashboard.UID] = dashboard
	dm.mu.Unlock()

	return dm.uploadDashboard(dashboard)
}

// uploadDashboard uploads dashboard to Grafana
func (dm *DashboardManager) uploadDashboard(dashboard *Dashboard) error {
	payload := map[string]interface{}{
		"dashboard": dm.convertToGrafanaFormat(dashboard),
		"folderId":  dm.config.FolderID,
		"overwrite": true,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal dashboard: %w", err)
	}

	url := fmt.Sprintf("%s/api/dashboards/db", dm.config.GrafanaURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", dm.config.APIKey))
	req.Header.Set("Content-Type", "application/json")

	resp, err := dm.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to upload dashboard: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("grafana API error: %s", string(body))
	}

	return nil
}

// convertToGrafanaFormat converts internal dashboard to Grafana format
func (dm *DashboardManager) convertToGrafanaFormat(dashboard *Dashboard) map[string]interface{} {
	panels := make([]map[string]interface{}, len(dashboard.Panels))
	for i, panel := range dashboard.Panels {
		targets := make([]map[string]interface{}, len(panel.Targets))
		for j, target := range panel.Targets {
			targets[j] = map[string]interface{}{
				"refId":        target.RefID,
				"expr":         target.Expr,
				"legendFormat": target.LegendFormat,
			}
		}

		panels[i] = map[string]interface{}{
			"id":         panel.ID,
			"title":      panel.Title,
			"type":       panel.Type,
			"datasource": panel.Datasource,
			"targets":    targets,
			"gridPos": map[string]interface{}{
				"x": panel.GridPos.X,
				"y": panel.GridPos.Y,
				"w": panel.GridPos.W,
				"h": panel.GridPos.H,
			},
		}
	}

	return map[string]interface{}{
		"uid":         dashboard.UID,
		"title":       dashboard.Title,
		"description": dashboard.Description,
		"tags":        dashboard.Tags,
		"panels":      panels,
	}
}

// GetDashboard retrieves a dashboard
func (dm *DashboardManager) GetDashboard(uid string) (*Dashboard, bool) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	dashboard, ok := dm.dashboards[uid]
	return dashboard, ok
}
