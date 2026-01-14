package dashboard

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// DashboardEngine is the core engine for managing dashboards
type DashboardEngine struct {
	// Storage
	dashboards map[string]*Dashboard
	templates  map[string]*DashboardTemplate
	snapshots  map[string]*DashboardSnapshot

	// Services
	widgetService   WidgetService
	renderer        DashboardRenderer
	validator       DashboardValidator
	eventHandlers   []DashboardEventHandler

	// Real-time updates
	subscribers    map[string][]chan *DashboardUpdate
	dataStreams    map[string]chan *WidgetData
	refreshWorkers map[string]*refreshWorker

	// Configuration
	config *EngineConfig

	// Concurrency control
	mutex           sync.RWMutex
	subscriberMutex sync.RWMutex
	streamMutex     sync.RWMutex
	workerMutex     sync.RWMutex

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// EngineConfig represents the configuration for the dashboard engine
type EngineConfig struct {
	// Performance settings
	MaxDashboards         int           `json:"max_dashboards"`
	MaxWidgetsPerDashboard int          `json:"max_widgets_per_dashboard"`
	MaxSubscribers        int           `json:"max_subscribers"`
	DataRetentionPeriod   time.Duration `json:"data_retention_period"`
	
	// Refresh settings
	DefaultRefreshInterval time.Duration `json:"default_refresh_interval"`
	MinRefreshInterval     time.Duration `json:"min_refresh_interval"`
	MaxRefreshInterval     time.Duration `json:"max_refresh_interval"`
	
	// Storage settings
	PersistenceEnabled   bool   `json:"persistence_enabled"`
	StorageBackend      string `json:"storage_backend"`
	ConnectionString    string `json:"connection_string"`
	
	// Security settings
	EnablePermissions    bool     `json:"enable_permissions"`
	AllowedDomains      []string `json:"allowed_domains"`
	RequireAuthentication bool    `json:"require_authentication"`
	
	// Performance optimization
	EnableCaching       bool          `json:"enable_caching"`
	CacheTTL           time.Duration `json:"cache_ttl"`
	EnableCompression  bool          `json:"enable_compression"`
}

// DefaultEngineConfig returns a default engine configuration
func DefaultEngineConfig() *EngineConfig {
	return &EngineConfig{
		MaxDashboards:          1000,
		MaxWidgetsPerDashboard: 50,
		MaxSubscribers:         100,
		DataRetentionPeriod:    24 * time.Hour,
		DefaultRefreshInterval: 30 * time.Second,
		MinRefreshInterval:     1 * time.Second,
		MaxRefreshInterval:     1 * time.Hour,
		PersistenceEnabled:     true,
		StorageBackend:         "memory",
		EnablePermissions:      true,
		RequireAuthentication:  true,
		EnableCaching:          true,
		CacheTTL:              5 * time.Minute,
		EnableCompression:      true,
	}
}

// refreshWorker manages automatic refresh for a dashboard
type refreshWorker struct {
	dashboardID string
	interval    time.Duration
	ticker      *time.Ticker
	done        chan struct{}
	engine      *DashboardEngine
}

// NewDashboardEngine creates a new dashboard engine
func NewDashboardEngine(config *EngineConfig) *DashboardEngine {
	if config == nil {
		config = DefaultEngineConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	engine := &DashboardEngine{
		dashboards:     make(map[string]*Dashboard),
		templates:      make(map[string]*DashboardTemplate),
		snapshots:      make(map[string]*DashboardSnapshot),
		subscribers:    make(map[string][]chan *DashboardUpdate),
		dataStreams:    make(map[string]chan *WidgetData),
		refreshWorkers: make(map[string]*refreshWorker),
		config:         config,
		ctx:            ctx,
		cancel:         cancel,
	}

	// Initialize default templates
	engine.loadDefaultTemplates()

	return engine
}

// Start starts the dashboard engine
func (e *DashboardEngine) Start() error {
	log.Println("Starting Dashboard Engine...")

	// Start background workers
	e.wg.Add(1)
	go e.maintenanceWorker()

	log.Println("Dashboard Engine started successfully")
	return nil
}

// Stop stops the dashboard engine
func (e *DashboardEngine) Stop() error {
	log.Println("Stopping Dashboard Engine...")

	e.cancel()
	e.wg.Wait()

	// Stop all refresh workers
	e.workerMutex.Lock()
	for _, worker := range e.refreshWorkers {
		worker.stop()
	}
	e.refreshWorkers = make(map[string]*refreshWorker)
	e.workerMutex.Unlock()

	// Close all streams
	e.streamMutex.Lock()
	for _, stream := range e.dataStreams {
		close(stream)
	}
	e.dataStreams = make(map[string]chan *WidgetData)
	e.streamMutex.Unlock()

	// Close all subscriber channels
	e.subscriberMutex.Lock()
	for _, subscribers := range e.subscribers {
		for _, ch := range subscribers {
			close(ch)
		}
	}
	e.subscribers = make(map[string][]chan *DashboardUpdate)
	e.subscriberMutex.Unlock()

	log.Println("Dashboard Engine stopped successfully")
	return nil
}

// SetWidgetService sets the widget service
func (e *DashboardEngine) SetWidgetService(service WidgetService) {
	e.widgetService = service
}

// SetRenderer sets the dashboard renderer
func (e *DashboardEngine) SetRenderer(renderer DashboardRenderer) {
	e.renderer = renderer
}

// SetValidator sets the dashboard validator
func (e *DashboardEngine) SetValidator(validator DashboardValidator) {
	e.validator = validator
}

// AddEventHandler adds an event handler
func (e *DashboardEngine) AddEventHandler(handler DashboardEventHandler) {
	e.eventHandlers = append(e.eventHandlers, handler)
}

// CreateDashboard creates a new dashboard
func (e *DashboardEngine) CreateDashboard(ctx context.Context, dashboard *Dashboard) (*Dashboard, error) {
	if dashboard == nil {
		return nil, fmt.Errorf("dashboard cannot be nil")
	}

	// Validate dashboard
	if e.validator != nil {
		if err := e.validator.ValidateDashboard(dashboard); err != nil {
			return nil, fmt.Errorf("dashboard validation failed: %w", err)
		}
	}

	// Check limits
	e.mutex.RLock()
	if len(e.dashboards) >= e.config.MaxDashboards {
		e.mutex.RUnlock()
		return nil, fmt.Errorf("maximum number of dashboards reached (%d)", e.config.MaxDashboards)
	}
	e.mutex.RUnlock()

	// Set default values
	if dashboard.ID == "" {
		dashboard.ID = uuid.New().String()
	}
	if dashboard.Version == "" {
		dashboard.Version = "1.0.0"
	}
	
	now := time.Now()
	dashboard.CreatedAt = now
	dashboard.UpdatedAt = now

	// Set default refresh rate if not specified
	if dashboard.RefreshRate == "" {
		dashboard.RefreshRate = RefreshInterval30Second
	}

	// Set default layout if not specified
	if dashboard.Layout.Type == "" {
		dashboard.Layout = Layout{
			Type:       "grid",
			Columns:    12,
			Rows:       24,
			CellWidth:  100,
			CellHeight: 50,
			Padding:    8,
			Margin:     8,
		}
	}

	// Validate widgets
	if len(dashboard.Widgets) > e.config.MaxWidgetsPerDashboard {
		return nil, fmt.Errorf("maximum number of widgets per dashboard reached (%d)", e.config.MaxWidgetsPerDashboard)
	}

	for i := range dashboard.Widgets {
		if dashboard.Widgets[i].ID == "" {
			dashboard.Widgets[i].ID = uuid.New().String()
		}
		
		if e.validator != nil {
			if err := e.validator.ValidateWidget(&dashboard.Widgets[i]); err != nil {
				return nil, fmt.Errorf("widget validation failed: %w", err)
			}
		}
	}

	// Store dashboard
	e.mutex.Lock()
	e.dashboards[dashboard.ID] = dashboard
	e.mutex.Unlock()

	// Start refresh worker if auto-refresh is enabled
	if dashboard.AutoRefresh {
		e.startRefreshWorker(dashboard.ID, dashboard.RefreshRate)
	}

	// Emit event
	e.emitEvent(&DashboardEvent{
		ID:          uuid.New().String(),
		Type:        EventDashboardCreated,
		DashboardID: dashboard.ID,
		TenantID:    dashboard.TenantID,
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"dashboard_name": dashboard.Name,
			"dashboard_type": dashboard.Type,
		},
	})

	return dashboard, nil
}

// GetDashboard retrieves a dashboard by ID
func (e *DashboardEngine) GetDashboard(ctx context.Context, id string, tenantID string) (*Dashboard, error) {
	if id == "" {
		return nil, fmt.Errorf("dashboard ID cannot be empty")
	}

	e.mutex.RLock()
	dashboard, exists := e.dashboards[id]
	e.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("dashboard with ID %s not found", id)
	}

	// Check tenant access
	if dashboard.TenantID != "" && dashboard.TenantID != tenantID && !dashboard.IsPublic {
		return nil, fmt.Errorf("access denied to dashboard %s", id)
	}

	// Emit view event
	e.emitEvent(&DashboardEvent{
		ID:          uuid.New().String(),
		Type:        EventDashboardViewed,
		DashboardID: dashboard.ID,
		TenantID:    tenantID,
		Timestamp:   time.Now(),
	})

	// Return a copy to prevent modification
	dashboardCopy := *dashboard
	return &dashboardCopy, nil
}

// UpdateDashboard updates an existing dashboard
func (e *DashboardEngine) UpdateDashboard(ctx context.Context, dashboard *Dashboard) (*Dashboard, error) {
	if dashboard == nil {
		return nil, fmt.Errorf("dashboard cannot be nil")
	}
	
	if dashboard.ID == "" {
		return nil, fmt.Errorf("dashboard ID cannot be empty")
	}

	// Validate dashboard
	if e.validator != nil {
		if err := e.validator.ValidateDashboard(dashboard); err != nil {
			return nil, fmt.Errorf("dashboard validation failed: %w", err)
		}
	}

	e.mutex.Lock()
	existingDashboard, exists := e.dashboards[dashboard.ID]
	if !exists {
		e.mutex.Unlock()
		return nil, fmt.Errorf("dashboard with ID %s not found", dashboard.ID)
	}

	// Preserve creation metadata
	dashboard.CreatedAt = existingDashboard.CreatedAt
	dashboard.CreatedBy = existingDashboard.CreatedBy
	dashboard.UpdatedAt = time.Now()

	// Validate widgets
	if len(dashboard.Widgets) > e.config.MaxWidgetsPerDashboard {
		e.mutex.Unlock()
		return nil, fmt.Errorf("maximum number of widgets per dashboard reached (%d)", e.config.MaxWidgetsPerDashboard)
	}

	for i := range dashboard.Widgets {
		if dashboard.Widgets[i].ID == "" {
			dashboard.Widgets[i].ID = uuid.New().String()
		}
		
		if e.validator != nil {
			if err := e.validator.ValidateWidget(&dashboard.Widgets[i]); err != nil {
				e.mutex.Unlock()
				return nil, fmt.Errorf("widget validation failed: %w", err)
			}
		}
	}

	e.dashboards[dashboard.ID] = dashboard
	e.mutex.Unlock()

	// Update refresh worker
	refreshRateChanged := existingDashboard.RefreshRate != dashboard.RefreshRate
	autoRefreshChanged := existingDashboard.AutoRefresh != dashboard.AutoRefresh

	if refreshRateChanged || autoRefreshChanged {
		e.stopRefreshWorker(dashboard.ID)
		if dashboard.AutoRefresh {
			e.startRefreshWorker(dashboard.ID, dashboard.RefreshRate)
		}
	}

	// Emit event
	e.emitEvent(&DashboardEvent{
		ID:          uuid.New().String(),
		Type:        EventDashboardUpdated,
		DashboardID: dashboard.ID,
		TenantID:    dashboard.TenantID,
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"dashboard_name": dashboard.Name,
		},
	})

	// Notify subscribers of the update
	e.notifySubscribers(dashboard.ID, &DashboardUpdate{
		Type:      "dashboard_config",
		Data:      dashboard,
		Timestamp: time.Now(),
	})

	return dashboard, nil
}

// DeleteDashboard deletes a dashboard
func (e *DashboardEngine) DeleteDashboard(ctx context.Context, id string, tenantID string) error {
	if id == "" {
		return fmt.Errorf("dashboard ID cannot be empty")
	}

	e.mutex.Lock()
	dashboard, exists := e.dashboards[id]
	if !exists {
		e.mutex.Unlock()
		return fmt.Errorf("dashboard with ID %s not found", id)
	}

	// Check tenant access
	if dashboard.TenantID != "" && dashboard.TenantID != tenantID && !dashboard.IsPublic {
		e.mutex.Unlock()
		return fmt.Errorf("access denied to dashboard %s", id)
	}

	delete(e.dashboards, id)
	e.mutex.Unlock()

	// Stop refresh worker
	e.stopRefreshWorker(id)

	// Clean up subscribers
	e.subscriberMutex.Lock()
	if subscribers, exists := e.subscribers[id]; exists {
		for _, ch := range subscribers {
			close(ch)
		}
		delete(e.subscribers, id)
	}
	e.subscriberMutex.Unlock()

	// Clean up data streams
	e.streamMutex.Lock()
	for streamKey := range e.dataStreams {
		if len(streamKey) > len(id) && streamKey[:len(id)] == id {
			close(e.dataStreams[streamKey])
			delete(e.dataStreams, streamKey)
		}
	}
	e.streamMutex.Unlock()

	// Emit event
	e.emitEvent(&DashboardEvent{
		ID:          uuid.New().String(),
		Type:        EventDashboardDeleted,
		DashboardID: id,
		TenantID:    tenantID,
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"dashboard_name": dashboard.Name,
		},
	})

	return nil
}

// ListDashboards lists all dashboards for a tenant
func (e *DashboardEngine) ListDashboards(ctx context.Context, tenantID string, filters map[string]interface{}) ([]*Dashboard, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	var result []*Dashboard
	for _, dashboard := range e.dashboards {
		// Check tenant access
		if dashboard.TenantID != "" && dashboard.TenantID != tenantID && !dashboard.IsPublic {
			continue
		}

		// Apply filters
		if e.matchesFilters(dashboard, filters) {
			// Return a copy
			dashboardCopy := *dashboard
			result = append(result, &dashboardCopy)
		}
	}

	return result, nil
}

// Subscribe subscribes to dashboard updates
func (e *DashboardEngine) Subscribe(ctx context.Context, dashboardID string) (<-chan *DashboardUpdate, error) {
	if dashboardID == "" {
		return nil, fmt.Errorf("dashboard ID cannot be empty")
	}

	// Check if dashboard exists
	e.mutex.RLock()
	_, exists := e.dashboards[dashboardID]
	e.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("dashboard with ID %s not found", dashboardID)
	}

	// Create subscriber channel
	updatesCh := make(chan *DashboardUpdate, 100)

	e.subscriberMutex.Lock()
	if len(e.subscribers[dashboardID]) >= e.config.MaxSubscribers {
		e.subscriberMutex.Unlock()
		close(updatesCh)
		return nil, fmt.Errorf("maximum number of subscribers reached for dashboard %s", dashboardID)
	}

	e.subscribers[dashboardID] = append(e.subscribers[dashboardID], updatesCh)
	e.subscriberMutex.Unlock()

	// Handle context cancellation
	go func() {
		<-ctx.Done()
		e.unsubscribe(dashboardID, updatesCh)
	}()

	return updatesCh, nil
}

// GetWidgetData gets data for a specific widget
func (e *DashboardEngine) GetWidgetData(ctx context.Context, dashboardID string, widgetID string, timeRange TimeRange) (*WidgetData, error) {
	if e.widgetService == nil {
		return nil, fmt.Errorf("widget service not configured")
	}

	// Get dashboard
	dashboard, err := e.GetDashboard(ctx, dashboardID, "")
	if err != nil {
		return nil, err
	}

	// Find widget
	var widget *Widget
	for i := range dashboard.Widgets {
		if dashboard.Widgets[i].ID == widgetID {
			widget = &dashboard.Widgets[i]
			break
		}
	}

	if widget == nil {
		return nil, fmt.Errorf("widget with ID %s not found in dashboard %s", widgetID, dashboardID)
	}

	// Query widget data
	return e.widgetService.QueryWidgetData(ctx, widget, timeRange)
}

// RefreshDashboard refreshes all widgets in a dashboard
func (e *DashboardEngine) RefreshDashboard(ctx context.Context, dashboardID string) error {
	dashboard, err := e.GetDashboard(ctx, dashboardID, "")
	if err != nil {
		return err
	}

	// Refresh each widget
	for _, widget := range dashboard.Widgets {
		go func(w Widget) {
			widgetData, err := e.widgetService.QueryWidgetData(ctx, &w, dashboard.TimeRange)
			if err != nil {
				log.Printf("Error refreshing widget %s: %v", w.ID, err)
				return
			}

			// Notify subscribers
			e.notifySubscribers(dashboardID, &DashboardUpdate{
				Type:      "widget_data",
				WidgetID:  w.ID,
				Data:      widgetData,
				Timestamp: time.Now(),
			})
		}(widget)
	}

	// Emit event
	e.emitEvent(&DashboardEvent{
		ID:          uuid.New().String(),
		Type:        EventDataRefreshed,
		DashboardID: dashboardID,
		Timestamp:   time.Now(),
	})

	return nil
}

// Helper methods

func (e *DashboardEngine) startRefreshWorker(dashboardID string, interval RefreshInterval) {
	duration := e.parseRefreshInterval(interval)
	if duration < e.config.MinRefreshInterval {
		duration = e.config.MinRefreshInterval
	}
	if duration > e.config.MaxRefreshInterval {
		duration = e.config.MaxRefreshInterval
	}

	worker := &refreshWorker{
		dashboardID: dashboardID,
		interval:    duration,
		ticker:      time.NewTicker(duration),
		done:        make(chan struct{}),
		engine:      e,
	}

	e.workerMutex.Lock()
	e.refreshWorkers[dashboardID] = worker
	e.workerMutex.Unlock()

	go worker.run()
}

func (e *DashboardEngine) stopRefreshWorker(dashboardID string) {
	e.workerMutex.Lock()
	worker, exists := e.refreshWorkers[dashboardID]
	if exists {
		worker.stop()
		delete(e.refreshWorkers, dashboardID)
	}
	e.workerMutex.Unlock()
}

func (e *DashboardEngine) parseRefreshInterval(interval RefreshInterval) time.Duration {
	switch interval {
	case RefreshInterval1Second:
		return 1 * time.Second
	case RefreshInterval5Second:
		return 5 * time.Second
	case RefreshInterval30Second:
		return 30 * time.Second
	case RefreshInterval1Minute:
		return 1 * time.Minute
	case RefreshInterval5Minute:
		return 5 * time.Minute
	case RefreshInterval15Minute:
		return 15 * time.Minute
	case RefreshInterval1Hour:
		return 1 * time.Hour
	default:
		return 30 * time.Second
	}
}

func (e *DashboardEngine) notifySubscribers(dashboardID string, update *DashboardUpdate) {
	e.subscriberMutex.RLock()
	subscribers, exists := e.subscribers[dashboardID]
	e.subscriberMutex.RUnlock()

	if !exists {
		return
	}

	for _, ch := range subscribers {
		select {
		case ch <- update:
		default:
			// Channel is full, skip this subscriber
		}
	}
}

func (e *DashboardEngine) unsubscribe(dashboardID string, ch chan *DashboardUpdate) {
	e.subscriberMutex.Lock()
	defer e.subscriberMutex.Unlock()

	subscribers, exists := e.subscribers[dashboardID]
	if !exists {
		return
	}

	// Remove the channel from subscribers
	for i, subscriber := range subscribers {
		if subscriber == ch {
			e.subscribers[dashboardID] = append(subscribers[:i], subscribers[i+1:]...)
			close(ch)
			break
		}
	}

	// Clean up empty subscriber lists
	if len(e.subscribers[dashboardID]) == 0 {
		delete(e.subscribers, dashboardID)
	}
}

func (e *DashboardEngine) matchesFilters(dashboard *Dashboard, filters map[string]interface{}) bool {
	for key, value := range filters {
		switch key {
		case "type":
			if dashboard.Type != DashboardType(value.(string)) {
				return false
			}
		case "tag":
			found := false
			for _, tag := range dashboard.Tags {
				if tag == value.(string) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		case "created_by":
			if dashboard.CreatedBy != value.(string) {
				return false
			}
		}
	}
	return true
}

func (e *DashboardEngine) emitEvent(event *DashboardEvent) {
	for _, handler := range e.eventHandlers {
		go func(h DashboardEventHandler) {
			if err := h.HandleEvent(context.Background(), event); err != nil {
				log.Printf("Error handling dashboard event: %v", err)
			}
		}(handler)
	}
}

func (e *DashboardEngine) loadDefaultTemplates() {
	// Load default dashboard templates
	templates := []*DashboardTemplate{
		{
			ID:          "system-overview",
			Name:        "System Overview",
			Description: "High-level system monitoring dashboard",
			Category:    "operations",
			Type:        DashboardTypeOperations,
			IsOfficial:  true,
			Template:    *e.createSystemOverviewTemplate(),
			CreatedAt:   time.Now(),
		},
		{
			ID:          "vm-management",
			Name:        "VM Management",
			Description: "Virtual machine monitoring and management",
			Category:    "vm",
			Type:        DashboardTypeOperations,
			IsOfficial:  true,
			Template:    *e.createVMManagementTemplate(),
			CreatedAt:   time.Now(),
		},
	}

	for _, template := range templates {
		e.templates[template.ID] = template
	}
}

func (e *DashboardEngine) createSystemOverviewTemplate() *Dashboard {
	return &Dashboard{
		Name:        "System Overview",
		Type:        DashboardTypeOperations,
		RefreshRate: RefreshInterval30Second,
		AutoRefresh: true,
		Layout: Layout{
			Type:       "grid",
			Columns:    12,
			Rows:       24,
			CellWidth:  100,
			CellHeight: 50,
			Padding:    8,
			Margin:     8,
		},
		Widgets: []Widget{
			{
				ID:    "cpu-usage",
				Type:  "gauge",
				Title: "CPU Usage",
				Position: Position{X: 0, Y: 0},
				Size:     Size{Width: 3, Height: 3},
				Config: map[string]interface{}{
					"unit":      "percent",
					"min":       0,
					"max":       100,
					"threshold": 80,
				},
			},
			{
				ID:    "memory-usage",
				Type:  "gauge",
				Title: "Memory Usage",
				Position: Position{X: 3, Y: 0},
				Size:     Size{Width: 3, Height: 3},
				Config: map[string]interface{}{
					"unit":      "percent",
					"min":       0,
					"max":       100,
					"threshold": 85,
				},
			},
		},
	}
}

func (e *DashboardEngine) createVMManagementTemplate() *Dashboard {
	return &Dashboard{
		Name:        "VM Management",
		Type:        DashboardTypeOperations,
		RefreshRate: RefreshInterval5Second,
		AutoRefresh: true,
		Layout: Layout{
			Type:       "grid",
			Columns:    12,
			Rows:       24,
			CellWidth:  100,
			CellHeight: 50,
			Padding:    8,
			Margin:     8,
		},
		Widgets: []Widget{
			{
				ID:    "vm-count",
				Type:  "stat",
				Title: "Total VMs",
				Position: Position{X: 0, Y: 0},
				Size:     Size{Width: 2, Height: 2},
			},
			{
				ID:    "vm-status",
				Type:  "table",
				Title: "VM Status",
				Position: Position{X: 0, Y: 2},
				Size:     Size{Width: 6, Height: 8},
			},
		},
	}
}

func (e *DashboardEngine) maintenanceWorker() {
	defer e.wg.Done()

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.performMaintenance()
		}
	}
}

func (e *DashboardEngine) performMaintenance() {
	// Clean up expired snapshots
	cutoff := time.Now()
	
	e.mutex.Lock()
	for id, snapshot := range e.snapshots {
		if !snapshot.ExpiresAt.IsZero() && snapshot.ExpiresAt.Before(cutoff) {
			delete(e.snapshots, id)
		}
	}
	e.mutex.Unlock()

	log.Println("Dashboard maintenance completed")
}

// refreshWorker methods

func (w *refreshWorker) run() {
	for {
		select {
		case <-w.done:
			w.ticker.Stop()
			return
		case <-w.ticker.C:
			if err := w.engine.RefreshDashboard(context.Background(), w.dashboardID); err != nil {
				log.Printf("Error refreshing dashboard %s: %v", w.dashboardID, err)
			}
		}
	}
}

func (w *refreshWorker) stop() {
	close(w.done)
}