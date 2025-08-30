package dashboard

import (
	"context"
	"encoding/json"
	"time"
)

// DashboardType represents the type of dashboard
type DashboardType string

const (
	DashboardTypeExecutive   DashboardType = "executive"
	DashboardTypeOperations  DashboardType = "operations"
	DashboardTypeDeveloper   DashboardType = "developer"
	DashboardTypeSecurity    DashboardType = "security"
	DashboardTypeTenant      DashboardType = "tenant"
	DashboardTypeCustom      DashboardType = "custom"
)

// RefreshInterval represents dashboard refresh intervals
type RefreshInterval string

const (
	RefreshInterval1Second  RefreshInterval = "1s"
	RefreshInterval5Second  RefreshInterval = "5s"
	RefreshInterval30Second RefreshInterval = "30s"
	RefreshInterval1Minute  RefreshInterval = "1m"
	RefreshInterval5Minute  RefreshInterval = "5m"
	RefreshInterval15Minute RefreshInterval = "15m"
	RefreshInterval1Hour    RefreshInterval = "1h"
)

// Permission represents dashboard access permissions
type Permission string

const (
	PermissionRead   Permission = "read"
	PermissionWrite  Permission = "write"
	PermissionAdmin  Permission = "admin"
	PermissionDelete Permission = "delete"
)

// Dashboard represents a monitoring dashboard
type Dashboard struct {
	// Basic information
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Type        DashboardType `json:"type"`
	Version     string        `json:"version"`

	// Metadata
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	CreatedBy   string    `json:"created_by"`
	UpdatedBy   string    `json:"updated_by"`
	Tags        []string  `json:"tags"`

	// Layout and configuration
	Layout         Layout          `json:"layout"`
	Widgets        []Widget        `json:"widgets"`
	RefreshRate    RefreshInterval `json:"refresh_rate"`
	TimeRange      TimeRange       `json:"time_range"`
	AutoRefresh    bool            `json:"auto_refresh"`
	Theme          string          `json:"theme"`

	// Multi-tenancy and access control
	TenantID    string                `json:"tenant_id"`
	Permissions map[string]Permission `json:"permissions"`
	IsPublic    bool                  `json:"is_public"`
	IsDefault   bool                  `json:"is_default"`

	// Advanced features
	Variables   map[string]Variable `json:"variables"`
	Annotations []Annotation        `json:"annotations"`
	Alerts      []string            `json:"alerts"` // Alert IDs linked to this dashboard
	
	// Mobile responsiveness
	MobileLayout *Layout `json:"mobile_layout,omitempty"`
	IsResponsive bool    `json:"is_responsive"`
}

// Layout represents the dashboard layout configuration
type Layout struct {
	Type         string     `json:"type"` // "grid", "flex", "absolute"
	Columns      int        `json:"columns"`
	Rows         int        `json:"rows"`
	CellWidth    int        `json:"cell_width"`
	CellHeight   int        `json:"cell_height"`
	Padding      int        `json:"padding"`
	Margin       int        `json:"margin"`
	Breakpoints  Breakpoint `json:"breakpoints"`
}

// Breakpoint defines responsive breakpoints
type Breakpoint struct {
	Mobile  int `json:"mobile"`  // < 768px
	Tablet  int `json:"tablet"`  // 768px - 1024px
	Desktop int `json:"desktop"` // > 1024px
}

// Widget represents a dashboard widget
type Widget struct {
	// Basic information
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Title       string    `json:"title"`
	Description string    `json:"description"`

	// Position and size
	Position Position `json:"position"`
	Size     Size     `json:"size"`
	ZIndex   int      `json:"z_index"`

	// Configuration
	Config      map[string]interface{} `json:"config"`
	DataSources []DataSource          `json:"data_sources"`
	Queries     []Query               `json:"queries"`

	// Styling and behavior
	Style       WidgetStyle `json:"style"`
	Interactions []Interaction `json:"interactions"`

	// Responsive behavior
	ResponsiveConfig map[string]interface{} `json:"responsive_config"`
	HideOnMobile     bool                  `json:"hide_on_mobile"`
}

// Position represents widget position
type Position struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// Size represents widget size
type Size struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// WidgetStyle represents widget styling
type WidgetStyle struct {
	BackgroundColor string            `json:"background_color"`
	BorderColor     string            `json:"border_color"`
	BorderWidth     int               `json:"border_width"`
	BorderRadius    int               `json:"border_radius"`
	Padding         int               `json:"padding"`
	Margin          int               `json:"margin"`
	FontSize        string            `json:"font_size"`
	FontWeight      string            `json:"font_weight"`
	Custom          map[string]string `json:"custom"`
}

// Interaction represents widget interactions
type Interaction struct {
	Type   string                 `json:"type"`   // "click", "hover", "drill_down"
	Action string                 `json:"action"` // "navigate", "filter", "alert"
	Config map[string]interface{} `json:"config"`
}

// DataSource represents a data source for widgets
type DataSource struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"` // "prometheus", "elasticsearch", "custom"
	URL      string                 `json:"url"`
	Database string                 `json:"database"`
	Config   map[string]interface{} `json:"config"`
}

// Query represents a data query
type Query struct {
	ID         string                 `json:"id"`
	DataSource string                 `json:"data_source"`
	QueryText  string                 `json:"query_text"`
	Interval   string                 `json:"interval"`
	TimeRange  *TimeRange             `json:"time_range,omitempty"`
	Variables  map[string]interface{} `json:"variables"`
}

// TimeRange represents a time range for queries
type TimeRange struct {
	From string `json:"from"` // "now-1h", "2023-01-01T00:00:00Z"
	To   string `json:"to"`   // "now", "2023-01-02T00:00:00Z"
}

// Variable represents a dashboard variable
type Variable struct {
	Name         string      `json:"name"`
	Label        string      `json:"label"`
	Type         string      `json:"type"` // "query", "custom", "constant", "datasource"
	Query        string      `json:"query"`
	Options      []Option    `json:"options"`
	DefaultValue interface{} `json:"default_value"`
	Multi        bool        `json:"multi"`
	Required     bool        `json:"required"`
}

// Option represents a variable option
type Option struct {
	Text     string      `json:"text"`
	Value    interface{} `json:"value"`
	Selected bool        `json:"selected"`
}

// Annotation represents a dashboard annotation
type Annotation struct {
	ID          string                 `json:"id"`
	Text        string                 `json:"text"`
	Time        time.Time              `json:"time"`
	Tags        []string               `json:"tags"`
	Type        string                 `json:"type"` // "info", "warning", "error", "custom"
	Color       string                 `json:"color"`
	Icon        string                 `json:"icon"`
	URL         string                 `json:"url"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// DashboardFilter represents filters applied to a dashboard
type DashboardFilter struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"` // "eq", "ne", "gt", "lt", "in", "contains"
	Value    interface{} `json:"value"`
}

// DashboardSnapshot represents a dashboard snapshot
type DashboardSnapshot struct {
	ID          string    `json:"id"`
	DashboardID string    `json:"dashboard_id"`
	Name        string    `json:"name"`
	Data        Dashboard `json:"data"`
	CreatedAt   time.Time `json:"created_at"`
	CreatedBy   string    `json:"created_by"`
	ExpiresAt   time.Time `json:"expires_at"`
	IsPublic    bool      `json:"is_public"`
}

// DashboardTemplate represents a reusable dashboard template
type DashboardTemplate struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Type        DashboardType          `json:"type"`
	Template    Dashboard              `json:"template"`
	Variables   map[string]interface{} `json:"variables"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	CreatedBy   string                 `json:"created_by"`
	IsOfficial  bool                   `json:"is_official"`
	Downloads   int                    `json:"downloads"`
	Rating      float64                `json:"rating"`
}

// WidgetData represents data for a widget
type WidgetData struct {
	WidgetID  string                   `json:"widget_id"`
	Data      []DataPoint              `json:"data"`
	Metadata  map[string]interface{}   `json:"metadata"`
	UpdatedAt time.Time                `json:"updated_at"`
	Status    string                   `json:"status"` // "loading", "success", "error"
	Error     string                   `json:"error,omitempty"`
}

// DataPoint represents a single data point
type DataPoint struct {
	Timestamp time.Time              `json:"timestamp"`
	Value     interface{}            `json:"value"`
	Labels    map[string]string      `json:"labels"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// DashboardUpdate represents an update to a dashboard
type DashboardUpdate struct {
	Type      string      `json:"type"` // "widget_data", "dashboard_config", "alert"
	WidgetID  string      `json:"widget_id,omitempty"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
}

// DashboardExport represents a dashboard export configuration
type DashboardExport struct {
	Format    string            `json:"format"` // "json", "yaml", "pdf"
	Dashboard Dashboard         `json:"dashboard"`
	Data      []WidgetData      `json:"data,omitempty"`
	Options   map[string]interface{} `json:"options"`
}

// DashboardImport represents a dashboard import configuration
type DashboardImport struct {
	Source      string                 `json:"source"` // "file", "url", "template"
	Data        json.RawMessage        `json:"data"`
	Options     map[string]interface{} `json:"options"`
	Overwrite   bool                   `json:"overwrite"`
	TenantID    string                 `json:"tenant_id"`
}

// DashboardService defines the interface for dashboard operations
type DashboardService interface {
	// Dashboard CRUD operations
	CreateDashboard(ctx context.Context, dashboard *Dashboard) (*Dashboard, error)
	GetDashboard(ctx context.Context, id string, tenantID string) (*Dashboard, error)
	UpdateDashboard(ctx context.Context, dashboard *Dashboard) (*Dashboard, error)
	DeleteDashboard(ctx context.Context, id string, tenantID string) error
	ListDashboards(ctx context.Context, tenantID string, filters map[string]interface{}) ([]*Dashboard, error)

	// Dashboard templates
	CreateTemplate(ctx context.Context, template *DashboardTemplate) (*DashboardTemplate, error)
	GetTemplate(ctx context.Context, id string) (*DashboardTemplate, error)
	ListTemplates(ctx context.Context, category string) ([]*DashboardTemplate, error)
	InstantiateTemplate(ctx context.Context, templateID string, variables map[string]interface{}, tenantID string) (*Dashboard, error)

	// Dashboard snapshots
	CreateSnapshot(ctx context.Context, dashboardID string, name string, expiresAt time.Time) (*DashboardSnapshot, error)
	GetSnapshot(ctx context.Context, id string) (*DashboardSnapshot, error)
	ListSnapshots(ctx context.Context, dashboardID string) ([]*DashboardSnapshot, error)

	// Widget data management
	GetWidgetData(ctx context.Context, dashboardID string, widgetID string, timeRange TimeRange) (*WidgetData, error)
	RefreshWidgetData(ctx context.Context, dashboardID string, widgetID string) (*WidgetData, error)
	StreamWidgetData(ctx context.Context, dashboardID string, widgetID string) (<-chan *WidgetData, error)

	// Export and import
	ExportDashboard(ctx context.Context, id string, format string, options map[string]interface{}) (*DashboardExport, error)
	ImportDashboard(ctx context.Context, importConfig *DashboardImport) (*Dashboard, error)

	// Permissions and sharing
	GrantPermission(ctx context.Context, dashboardID string, userID string, permission Permission) error
	RevokePermission(ctx context.Context, dashboardID string, userID string) error
	GetPermissions(ctx context.Context, dashboardID string) (map[string]Permission, error)
}

// WidgetService defines the interface for widget operations
type WidgetService interface {
	// Widget management
	CreateWidget(ctx context.Context, dashboardID string, widget *Widget) (*Widget, error)
	UpdateWidget(ctx context.Context, dashboardID string, widget *Widget) (*Widget, error)
	DeleteWidget(ctx context.Context, dashboardID string, widgetID string) error
	GetWidget(ctx context.Context, dashboardID string, widgetID string) (*Widget, error)

	// Widget data
	QueryWidgetData(ctx context.Context, widget *Widget, timeRange TimeRange) (*WidgetData, error)
	StreamWidgetUpdates(ctx context.Context, dashboardID string, widgetID string) (<-chan *DashboardUpdate, error)
}

// DashboardRenderer defines the interface for rendering dashboards
type DashboardRenderer interface {
	// Render dashboard
	RenderDashboard(ctx context.Context, dashboard *Dashboard, format string) ([]byte, error)
	RenderWidget(ctx context.Context, widget *Widget, data *WidgetData, format string) ([]byte, error)

	// Responsive rendering
	RenderResponsive(ctx context.Context, dashboard *Dashboard, deviceType string) ([]byte, error)

	// Export rendering
	RenderToPDF(ctx context.Context, dashboard *Dashboard, options map[string]interface{}) ([]byte, error)
	RenderToImage(ctx context.Context, dashboard *Dashboard, format string) ([]byte, error)
}

// DashboardValidator defines the interface for dashboard validation
type DashboardValidator interface {
	ValidateDashboard(dashboard *Dashboard) error
	ValidateWidget(widget *Widget) error
	ValidateQuery(query *Query) error
	ValidateDataSource(dataSource *DataSource) error
}

// EventType represents different types of dashboard events
type EventType string

const (
	EventDashboardCreated  EventType = "dashboard.created"
	EventDashboardUpdated  EventType = "dashboard.updated"
	EventDashboardDeleted  EventType = "dashboard.deleted"
	EventDashboardViewed   EventType = "dashboard.viewed"
	EventWidgetCreated     EventType = "widget.created"
	EventWidgetUpdated     EventType = "widget.updated"
	EventWidgetDeleted     EventType = "widget.deleted"
	EventDataRefreshed     EventType = "data.refreshed"
	EventAlertTriggered    EventType = "alert.triggered"
)

// DashboardEvent represents a dashboard-related event
type DashboardEvent struct {
	ID          string                 `json:"id"`
	Type        EventType              `json:"type"`
	DashboardID string                 `json:"dashboard_id"`
	WidgetID    string                 `json:"widget_id,omitempty"`
	UserID      string                 `json:"user_id"`
	TenantID    string                 `json:"tenant_id"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data"`
	Source      string                 `json:"source"`
}

// DashboardEventHandler defines the interface for handling dashboard events
type DashboardEventHandler interface {
	HandleEvent(ctx context.Context, event *DashboardEvent) error
}

// MetricCalculation represents calculations on metrics
type MetricCalculation struct {
	Type       string      `json:"type"`       // "sum", "avg", "min", "max", "count", "rate", "increase"
	Field      string      `json:"field"`
	GroupBy    []string    `json:"group_by"`
	Having     []Filter    `json:"having"`
	OrderBy    string      `json:"order_by"`
	Limit      int         `json:"limit"`
	Parameters interface{} `json:"parameters"`
}

// Filter represents a data filter
type Filter struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
	Logic    string      `json:"logic"` // "AND", "OR"
}

// AlertRule represents an alert rule linked to a dashboard
type AlertRule struct {
	ID          string    `json:"id"`
	DashboardID string    `json:"dashboard_id"`
	WidgetID    string    `json:"widget_id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Query       Query     `json:"query"`
	Condition   Condition `json:"condition"`
	Actions     []Action  `json:"actions"`
	Enabled     bool      `json:"enabled"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// Condition represents an alert condition
type Condition struct {
	Operator  string      `json:"operator"`  // "gt", "lt", "eq", "ne", "gte", "lte"
	Value     interface{} `json:"value"`
	Duration  string      `json:"duration"`  // "5m", "1h"
	Function  string      `json:"function"`  // "avg", "max", "min", "sum"
}

// Action represents an alert action
type Action struct {
	Type   string                 `json:"type"`   // "email", "webhook", "slack"
	Config map[string]interface{} `json:"config"`
}