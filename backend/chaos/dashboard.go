// Package chaos - Chaos engineering dashboard and monitoring
package chaos

import (
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// Dashboard provides web interface for chaos engineering
type Dashboard struct {
	engine    *ChaosEngine
	server    *http.Server
	upgrader  websocket.Upgrader
	clients   map[*websocket.Conn]bool
	broadcast chan interface{}
	logger    *zap.Logger
	mu        sync.RWMutex
}

// DashboardConfig configures the chaos dashboard
type DashboardConfig struct {
	Port           int      `json:"port"`
	EnableAuth     bool     `json:"enable_auth"`
	AllowedOrigins []string `json:"allowed_origins"`
	RefreshRate    int      `json:"refresh_rate"` // seconds
}

// NewDashboard creates a new chaos dashboard
func NewDashboard(engine *ChaosEngine, config *DashboardConfig, logger *zap.Logger) *Dashboard {
	return &Dashboard{
		engine: engine,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// In production, check against allowed origins
				return true
			},
		},
		clients:   make(map[*websocket.Conn]bool),
		broadcast: make(chan interface{}, 100),
		logger:    logger,
	}
}

// Start launches the dashboard server
func (d *Dashboard) Start(ctx context.Context) error {
	router := mux.NewRouter()
	
	// API endpoints
	api := router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/experiments", d.handleGetExperiments).Methods("GET")
	api.HandleFunc("/experiments", d.handleCreateExperiment).Methods("POST")
	api.HandleFunc("/experiments/{id}", d.handleGetExperiment).Methods("GET")
	api.HandleFunc("/experiments/{id}/run", d.handleRunExperiment).Methods("POST")
	api.HandleFunc("/experiments/{id}/stop", d.handleStopExperiment).Methods("POST")
	api.HandleFunc("/experiments/{id}/rollback", d.handleRollbackExperiment).Methods("POST")
	api.HandleFunc("/metrics", d.handleGetMetrics).Methods("GET")
	api.HandleFunc("/impact", d.handleGetImpact).Methods("GET")
	api.HandleFunc("/learnings", d.handleGetLearnings).Methods("GET")
	api.HandleFunc("/schedule", d.handleGetSchedule).Methods("GET")
	api.HandleFunc("/gamedays", d.handleGameDays).Methods("GET", "POST")
	
	// WebSocket for real-time updates
	router.HandleFunc("/ws", d.handleWebSocket)
	
	// Static files and UI
	router.HandleFunc("/", d.handleIndex)
	router.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir("./static"))))
	
	d.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", 8090),
		Handler: router,
	}
	
	// Start broadcast handler
	go d.broadcastHandler()
	
	// Start metrics pusher
	go d.metricsPusher(ctx)
	
	d.logger.Info("Starting chaos dashboard", zap.String("addr", d.server.Addr))
	
	go func() {
		if err := d.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			d.logger.Error("Dashboard server error", zap.Error(err))
		}
	}()
	
	return nil
}

// Stop gracefully stops the dashboard
func (d *Dashboard) Stop(ctx context.Context) error {
	return d.server.Shutdown(ctx)
}

// handleIndex serves the main dashboard HTML
func (d *Dashboard) handleIndex(w http.ResponseWriter, r *http.Request) {
	tmpl := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaCron Chaos Engineering Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 40px 0;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            opacity: 0.9;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }
        .experiments-section {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        .experiment-list {
            display: grid;
            gap: 15px;
            margin-top: 20px;
        }
        .experiment-card {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: 20px;
        }
        .experiment-info h3 {
            margin-bottom: 8px;
        }
        .experiment-meta {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-running { background: #48bb78; }
        .status-scheduled { background: #4299e1; }
        .status-completed { background: #9f7aea; }
        .status-failed { background: #f56565; }
        .control-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-danger {
            background: linear-gradient(135deg, #f56565, #c53030);
            color: white;
        }
        .btn-success {
            background: linear-gradient(135deg, #48bb78, #2f855a);
            color: white;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .impact-visualization {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        .blast-radius-chart {
            height: 300px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            opacity: 0.7;
        }
        .timeline {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 30px;
        }
        .timeline-item {
            position: relative;
            padding-left: 40px;
            margin-bottom: 20px;
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 5px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #667eea;
        }
        .timeline-item::after {
            content: '';
            position: absolute;
            left: 14px;
            top: 15px;
            width: 2px;
            height: calc(100% + 10px);
            background: rgba(255,255,255,0.2);
        }
        .timeline-item:last-child::after {
            display: none;
        }
        .ws-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.85em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ws-connected {
            background: #48bb78;
        }
        .ws-disconnected {
            background: #f56565;
        }
        .pulse {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.5); }
            100% { opacity: 1; transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌪️ NovaCron Chaos Engineering</h1>
            <div class="subtitle">Building Resilience Through Controlled Chaos</div>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Active Experiments</div>
                <div class="metric-value" id="active-experiments">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Resilience Score</div>
                <div class="metric-value" id="resilience-score">98.5%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recovery Time</div>
                <div class="metric-value" id="recovery-time">1.2s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Experiments Today</div>
                <div class="metric-value" id="experiments-today">12</div>
            </div>
        </div>
        
        <div class="control-panel">
            <h2 style="margin-bottom: 20px;">🎮 Chaos Control</h2>
            <button class="btn btn-primary" onclick="createExperiment()">Create Experiment</button>
            <button class="btn btn-success" onclick="scheduleGameDay()">Schedule Game Day</button>
            <button class="btn btn-danger" onclick="emergencyStop()">Emergency Stop All</button>
        </div>
        
        <div class="experiments-section">
            <h2>🧪 Active Experiments</h2>
            <div class="experiment-list" id="experiment-list">
                <!-- Experiments will be populated here -->
            </div>
        </div>
        
        <div class="impact-visualization">
            <h2 style="margin-bottom: 20px;">📊 Blast Radius Visualization</h2>
            <div class="blast-radius-chart">
                <canvas id="blast-chart"></canvas>
            </div>
        </div>
        
        <div class="timeline">
            <h2 style="margin-bottom: 20px;">📝 Recent Events</h2>
            <div id="timeline-events">
                <!-- Timeline events will be populated here -->
            </div>
        </div>
    </div>
    
    <div class="ws-status ws-disconnected" id="ws-status">
        <div class="pulse"></div>
        <span id="ws-text">Disconnected</span>
    </div>
    
    <script>
        let ws = null;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://' + window.location.host + '/ws');
            
            ws.onopen = function() {
                document.getElementById('ws-status').className = 'ws-status ws-connected';
                document.getElementById('ws-text').textContent = 'Connected';
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                document.getElementById('ws-status').className = 'ws-status ws-disconnected';
                document.getElementById('ws-text').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateDashboard(data) {
            if (data.type === 'metrics') {
                document.getElementById('active-experiments').textContent = data.active_experiments || 0;
                document.getElementById('resilience-score').textContent = (data.resilience_score * 100).toFixed(1) + '%';
                document.getElementById('recovery-time').textContent = data.recovery_time + 's';
                document.getElementById('experiments-today').textContent = data.experiments_today || 0;
            } else if (data.type === 'experiment_update') {
                updateExperimentList(data.experiments);
            } else if (data.type === 'event') {
                addTimelineEvent(data.event);
            }
        }
        
        function updateExperimentList(experiments) {
            const list = document.getElementById('experiment-list');
            list.innerHTML = experiments.map(exp => createExperimentCard(exp)).join('');
        }
        
        function createExperimentCard(exp) {
            return '<div class="experiment-card">' +
                '<div class="experiment-info">' +
                '<h3>' + exp.name + '</h3>' +
                '<div class="experiment-meta">' +
                '<span>Type: ' + exp.type + '</span>' +
                '<span>Blast: ' + exp.blast_radius.percentage + '%</span>' +
                '<span>Duration: ' + exp.safety.max_duration + '</span>' +
                '</div>' +
                '</div>' +
                '<span class="status-badge status-' + exp.status + '">' + exp.status + '</span>' +
                '</div>';
        }
        
        function addTimelineEvent(event) {
            const timeline = document.getElementById('timeline-events');
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = '<strong>' + event.timestamp + '</strong> - ' + event.description;
            timeline.insertBefore(item, timeline.firstChild);
            
            // Keep only last 10 events
            while (timeline.children.length > 10) {
                timeline.removeChild(timeline.lastChild);
            }
        }
        
        function createExperiment() {
            // Open experiment creation modal
            alert('Experiment creation UI would open here');
        }
        
        function scheduleGameDay() {
            // Open game day scheduling modal
            alert('Game day scheduling UI would open here');
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to stop all experiments?')) {
                fetch('/api/experiments/stop-all', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => console.log('All experiments stopped:', data));
            }
        }
        
        // Initialize
        connectWebSocket();
        
        // Fetch initial data
        fetch('/api/experiments')
            .then(response => response.json())
            .then(data => updateExperimentList(data));
    </script>
</body>
</html>`
	
	t, _ := template.New("dashboard").Parse(tmpl)
	t.Execute(w, nil)
}

// handleWebSocket manages WebSocket connections
func (d *Dashboard) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := d.upgrader.Upgrade(w, r, nil)
	if err != nil {
		d.logger.Error("WebSocket upgrade failed", zap.Error(err))
		return
	}
	
	d.mu.Lock()
	d.clients[conn] = true
	d.mu.Unlock()
	
	// Send initial state
	d.sendInitialState(conn)
	
	// Handle connection
	go d.handleConnection(conn)
}

// handleConnection manages individual WebSocket connections
func (d *Dashboard) handleConnection(conn *websocket.Conn) {
	defer func() {
		d.mu.Lock()
		delete(d.clients, conn)
		d.mu.Unlock()
		conn.Close()
	}()
	
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				d.logger.Error("WebSocket error", zap.Error(err))
			}
			break
		}
		
		// Handle client messages
		d.handleClientMessage(conn, msg)
	}
}

// broadcastHandler sends updates to all connected clients
func (d *Dashboard) broadcastHandler() {
	for {
		msg := <-d.broadcast
		d.mu.RLock()
		for client := range d.clients {
			err := client.WriteJSON(msg)
			if err != nil {
				client.Close()
			}
		}
		d.mu.RUnlock()
	}
}

// metricsPusher periodically pushes metrics to clients
func (d *Dashboard) metricsPusher(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := d.collectMetrics()
			d.broadcast <- map[string]interface{}{
				"type":    "metrics",
				"metrics": metrics,
			}
		}
	}
}

// collectMetrics gathers current metrics
func (d *Dashboard) collectMetrics() map[string]interface{} {
	d.engine.mu.RLock()
	activeCount := len(d.engine.activeExperiments)
	totalCount := len(d.engine.experiments)
	d.engine.mu.RUnlock()
	
	// Calculate resilience score
	resilienceScore := d.calculateResilienceScore()
	
	// Get recovery time
	recoveryTime := d.getAverageRecoveryTime()
	
	return map[string]interface{}{
		"active_experiments": activeCount,
		"total_experiments":  totalCount,
		"resilience_score":   resilienceScore,
		"recovery_time":      recoveryTime,
		"experiments_today":  d.getExperimentsToday(),
	}
}

// calculateResilienceScore computes system resilience score
func (d *Dashboard) calculateResilienceScore() float64 {
	// This would analyze experiment results
	// For now, return a sample score
	return 0.985
}

// getAverageRecoveryTime calculates average recovery time
func (d *Dashboard) getAverageRecoveryTime() float64 {
	// This would calculate from experiment results
	// For now, return a sample time
	return 1.2
}

// getExperimentsToday counts experiments run today
func (d *Dashboard) getExperimentsToday() int {
	count := 0
	today := time.Now().Truncate(24 * time.Hour)
	
	d.engine.mu.RLock()
	for _, exp := range d.engine.experiments {
		if exp.StartTime.After(today) {
			count++
		}
	}
	d.engine.mu.RUnlock()
	
	return count
}

// API Handlers

func (d *Dashboard) handleGetExperiments(w http.ResponseWriter, r *http.Request) {
	d.engine.mu.RLock()
	experiments := make([]*ChaosExperiment, 0, len(d.engine.experiments))
	for _, exp := range d.engine.experiments {
		experiments = append(experiments, exp)
	}
	d.engine.mu.RUnlock()
	
	// Sort by start time
	sort.Slice(experiments, func(i, j int) bool {
		return experiments[i].StartTime.After(experiments[j].StartTime)
	})
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(experiments)
}

func (d *Dashboard) handleCreateExperiment(w http.ResponseWriter, r *http.Request) {
	var spec ExperimentSpec
	if err := json.NewDecoder(r.Body).Decode(&spec); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	experiment, err := d.engine.CreateExperiment(&spec)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Broadcast update
	d.broadcast <- map[string]interface{}{
		"type":       "experiment_created",
		"experiment": experiment,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(experiment)
}

func (d *Dashboard) handleGetExperiment(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]
	
	d.engine.mu.RLock()
	experiment, exists := d.engine.experiments[id]
	d.engine.mu.RUnlock()
	
	if !exists {
		http.Error(w, "Experiment not found", http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(experiment)
}

func (d *Dashboard) handleRunExperiment(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]
	
	if err := d.engine.RunExperiment(context.Background(), id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Broadcast update
	d.broadcast <- map[string]interface{}{
		"type":         "experiment_started",
		"experiment_id": id,
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "started"})
}

func (d *Dashboard) handleStopExperiment(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]
	
	d.engine.mu.RLock()
	experiment, exists := d.engine.experiments[id]
	d.engine.mu.RUnlock()
	
	if !exists {
		http.Error(w, "Experiment not found", http.StatusNotFound)
		return
	}
	
	// Stop experiment
	experiment.Status = StatusAborted
	
	// Broadcast update
	d.broadcast <- map[string]interface{}{
		"type":         "experiment_stopped",
		"experiment_id": id,
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "stopped"})
}

func (d *Dashboard) handleRollbackExperiment(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]
	
	d.engine.mu.RLock()
	experiment, exists := d.engine.experiments[id]
	d.engine.mu.RUnlock()
	
	if !exists {
		http.Error(w, "Experiment not found", http.StatusNotFound)
		return
	}
	
	if err := d.engine.rollbackExperiment(context.Background(), experiment); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Broadcast update
	d.broadcast <- map[string]interface{}{
		"type":         "experiment_rolled_back",
		"experiment_id": id,
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "rolled_back"})
}

func (d *Dashboard) handleGetMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := d.collectMetrics()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (d *Dashboard) handleGetImpact(w http.ResponseWriter, r *http.Request) {
	// Aggregate impact across all active experiments
	impact := d.aggregateImpact()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(impact)
}

func (d *Dashboard) handleGetLearnings(w http.ResponseWriter, r *http.Request) {
	// Get all learnings from experiments
	learnings := d.getAllLearnings()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(learnings)
}

func (d *Dashboard) handleGetSchedule(w http.ResponseWriter, r *http.Request) {
	// Get scheduled experiments
	scheduled := d.getScheduledExperiments()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(scheduled)
}

func (d *Dashboard) handleGameDays(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		// Get game day schedule
		gameDays := d.getGameDays()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(gameDays)
	} else if r.Method == "POST" {
		// Schedule new game day
		var gameDay GameDay
		if err := json.NewDecoder(r.Body).Decode(&gameDay); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		if err := d.scheduleGameDay(&gameDay); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(gameDay)
	}
}

// Helper methods

func (d *Dashboard) sendInitialState(conn *websocket.Conn) {
	// Send current experiments
	d.engine.mu.RLock()
	experiments := make([]*ChaosExperiment, 0, len(d.engine.experiments))
	for _, exp := range d.engine.experiments {
		experiments = append(experiments, exp)
	}
	d.engine.mu.RUnlock()
	
	conn.WriteJSON(map[string]interface{}{
		"type":        "initial_state",
		"experiments": experiments,
		"metrics":     d.collectMetrics(),
	})
}

func (d *Dashboard) handleClientMessage(conn *websocket.Conn, msg map[string]interface{}) {
	// Handle various client commands
	command, ok := msg["command"].(string)
	if !ok {
		return
	}
	
	switch command {
	case "subscribe":
		// Subscribe to specific updates
	case "unsubscribe":
		// Unsubscribe from updates
	case "get_details":
		// Get detailed information
	}
}

func (d *Dashboard) aggregateImpact() *ImpactAnalysis {
	// Aggregate impact from all active experiments
	aggregated := &ImpactAnalysis{
		AffectedNodes:    []string{},
		AffectedServices: []string{},
	}
	
	d.engine.mu.RLock()
	for _, exp := range d.engine.activeExperiments {
		if exp.Impact != nil {
			aggregated.AffectedNodes = append(aggregated.AffectedNodes, exp.Impact.AffectedNodes...)
			aggregated.AffectedServices = append(aggregated.AffectedServices, exp.Impact.AffectedServices...)
			aggregated.ErrorRate += exp.Impact.ErrorRate
		}
	}
	d.engine.mu.RUnlock()
	
	if len(d.engine.activeExperiments) > 0 {
		aggregated.ErrorRate /= float64(len(d.engine.activeExperiments))
	}
	
	return aggregated
}

func (d *Dashboard) getAllLearnings() []Learning {
	learnings := []Learning{}
	
	d.engine.mu.RLock()
	for _, exp := range d.engine.experiments {
		if exp.Results != nil {
			learnings = append(learnings, exp.Results.Learnings...)
		}
	}
	d.engine.mu.RUnlock()
	
	return learnings
}

func (d *Dashboard) getScheduledExperiments() []*ChaosExperiment {
	scheduled := []*ChaosExperiment{}
	
	d.engine.mu.RLock()
	for _, exp := range d.engine.experiments {
		if exp.Status == StatusScheduled {
			scheduled = append(scheduled, exp)
		}
	}
	d.engine.mu.RUnlock()
	
	return scheduled
}

func (d *Dashboard) getGameDays() []GameDay {
	// This would fetch from storage
	return []GameDay{}
}

func (d *Dashboard) scheduleGameDay(gameDay *GameDay) error {
	// This would schedule a game day
	return nil
}