package edge

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
)

// EdgeAgentConfig contains configuration for edge agent
type EdgeAgentConfig struct {
	// Agent identification
	AgentID       string            `json:"agent_id"`
	EdgeName      string            `json:"edge_name"`
	Region        string            `json:"region"`
	Zone          string            `json:"zone,omitempty"`
	Tags          map[string]string `json:"tags,omitempty"`

	// Connectivity configuration
	CloudEndpoint       string        `json:"cloud_endpoint"`
	WebSocketPath       string        `json:"websocket_path"`
	ReconnectInterval   time.Duration `json:"reconnect_interval"`
	HeartbeatInterval   time.Duration `json:"heartbeat_interval"`
	EnableTLS           bool          `json:"enable_tls"`
	SkipTLSVerification bool          `json:"skip_tls_verification"`

	// Resource constraints
	MaxCPUPercent    float64 `json:"max_cpu_percent"`
	MaxMemoryPercent float64 `json:"max_memory_percent"`
	MaxDiskPercent   float64 `json:"max_disk_percent"`
	MinFreeMemoryGB  float64 `json:"min_free_memory_gb"`

	// Operational settings
	DataPath          string        `json:"data_path"`
	LogLevel          string        `json:"log_level"`
	MetricsInterval   time.Duration `json:"metrics_interval"`
	CacheSize         int64         `json:"cache_size"`
	OfflineThreshold  time.Duration `json:"offline_threshold"`
	MaxRetainedLogs   int           `json:"max_retained_logs"`
}

// DefaultEdgeAgentConfig returns default configuration
func DefaultEdgeAgentConfig() EdgeAgentConfig {
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "unknown"
	}

	return EdgeAgentConfig{
		AgentID:             generateUUID(),
		EdgeName:            fmt.Sprintf("edge-%s", hostname),
		Region:              "default",
		CloudEndpoint:       "ws://localhost:8091",
		WebSocketPath:       "/ws/edge",
		ReconnectInterval:   30 * time.Second,
		HeartbeatInterval:   10 * time.Second,
		EnableTLS:           false,
		SkipTLSVerification: true,
		MaxCPUPercent:       80.0,
		MaxMemoryPercent:    85.0,
		MaxDiskPercent:      90.0,
		MinFreeMemoryGB:     1.0,
		DataPath:            "/var/lib/novacron-edge",
		LogLevel:            "info",
		MetricsInterval:     5 * time.Second,
		CacheSize:           100 * 1024 * 1024, // 100MB
		OfflineThreshold:    60 * time.Second,
		MaxRetainedLogs:     1000,
		Tags:                make(map[string]string),
	}
}

// EdgeAgent represents a lightweight edge computing agent
type EdgeAgent struct {
	config EdgeAgentConfig

	// Connection management
	conn         *websocket.Conn
	connMutex    sync.RWMutex
	connected    bool
	lastSeen     time.Time
	disconnected chan struct{}

	// System monitoring
	systemMonitor *SystemMonitor
	metricsCache  *MetricsCache
	
	// Local services
	localCache      *LocalCache
	taskManager     *EdgeTaskManager
	resourceManager *EdgeResourceManager
	
	// State management
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
	running    bool
	runMutex   sync.RWMutex

	// Message handling
	messageHandlers map[string]MessageHandler
	pendingTasks    map[string]*EdgeTask
	taskMutex       sync.RWMutex

	// Offline capabilities
	offlineMode    bool
	offlineTasks   []*EdgeTask
	offlineMetrics []*MetricSnapshot
	offlineMutex   sync.RWMutex
}

// MessageHandler handles specific message types
type MessageHandler func(agent *EdgeAgent, message *Message) error

// Message represents a WebSocket message
type Message struct {
	Type      string                 `json:"type"`
	ID        string                 `json:"id,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// EdgeTask represents a task to be executed on the edge
type EdgeTask struct {
	ID            string                 `json:"id"`
	Type          string                 `json:"type"`
	Priority      int                    `json:"priority"`
	Data          map[string]interface{} `json:"data"`
	ResourceReqs  ResourceRequirements   `json:"resource_requirements"`
	Deadline      time.Time              `json:"deadline,omitempty"`
	Status        TaskStatus             `json:"status"`
	Result        map[string]interface{} `json:"result,omitempty"`
	Error         string                 `json:"error,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
	StartedAt     time.Time              `json:"started_at,omitempty"`
	CompletedAt   time.Time              `json:"completed_at,omitempty"`
	RetryCount    int                    `json:"retry_count"`
	MaxRetries    int                    `json:"max_retries"`
}

// ResourceRequirements defines resource needs for a task
type ResourceRequirements struct {
	CPUCores   float64 `json:"cpu_cores"`
	MemoryMB   int64   `json:"memory_mb"`
	DiskMB     int64   `json:"disk_mb"`
	NetworkMB  int64   `json:"network_mb,omitempty"`
	GPUMemoryMB int64  `json:"gpu_memory_mb,omitempty"`
}

// TaskStatus represents task execution status
type TaskStatus string

const (
	TaskStatusPending    TaskStatus = "pending"
	TaskStatusQueued     TaskStatus = "queued" 
	TaskStatusRunning    TaskStatus = "running"
	TaskStatusCompleted  TaskStatus = "completed"
	TaskStatusFailed     TaskStatus = "failed"
	TaskStatusCancelled  TaskStatus = "cancelled"
	TaskStatusTimeout    TaskStatus = "timeout"
)

// NewEdgeAgent creates a new edge agent
func NewEdgeAgent(config EdgeAgentConfig) *EdgeAgent {
	ctx, cancel := context.WithCancel(context.Background())

	agent := &EdgeAgent{
		config:          config,
		connected:       false,
		disconnected:    make(chan struct{}),
		ctx:             ctx,
		cancel:          cancel,
		messageHandlers: make(map[string]MessageHandler),
		pendingTasks:    make(map[string]*EdgeTask),
		offlineTasks:    make([]*EdgeTask, 0),
		offlineMetrics:  make([]*MetricSnapshot, 0),
	}

	// Initialize components
	agent.systemMonitor = NewSystemMonitor(config.MetricsInterval)
	agent.metricsCache = NewMetricsCache(config.CacheSize)
	agent.localCache = NewLocalCache(config.CacheSize)
	agent.taskManager = NewEdgeTaskManager(config.MaxRetainedLogs)
	agent.resourceManager = NewEdgeResourceManager(config)

	// Setup message handlers
	agent.setupMessageHandlers()

	return agent
}

// Start starts the edge agent
func (a *EdgeAgent) Start() error {
	a.runMutex.Lock()
	defer a.runMutex.Unlock()

	if a.running {
		return fmt.Errorf("edge agent already running")
	}

	log.Printf("Starting NovaCron Edge Agent %s", a.config.AgentID)

	// Create data directory
	if err := os.MkdirAll(a.config.DataPath, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	// Start system monitoring
	if err := a.systemMonitor.Start(); err != nil {
		return fmt.Errorf("failed to start system monitor: %v", err)
	}

	// Start local services
	if err := a.localCache.Start(); err != nil {
		return fmt.Errorf("failed to start local cache: %v", err)
	}

	if err := a.taskManager.Start(); err != nil {
		return fmt.Errorf("failed to start task manager: %v", err)
	}

	if err := a.resourceManager.Start(); err != nil {
		return fmt.Errorf("failed to start resource manager: %v", err)
	}

	a.running = true

	// Start background goroutines
	a.wg.Add(4)
	go a.connectionLoop()
	go a.heartbeatLoop()
	go a.metricsLoop()
	go a.taskProcessingLoop()

	log.Printf("Edge agent %s started successfully", a.config.EdgeName)
	return nil
}

// Stop stops the edge agent
func (a *EdgeAgent) Stop() error {
	a.runMutex.Lock()
	defer a.runMutex.Unlock()

	if !a.running {
		return nil
	}

	log.Printf("Stopping edge agent %s", a.config.EdgeName)

	a.running = false
	a.cancel()

	// Close connection
	a.connMutex.Lock()
	if a.conn != nil {
		a.conn.Close()
		a.conn = nil
	}
	a.connMutex.Unlock()

	// Wait for goroutines
	a.wg.Wait()

	// Stop services
	a.resourceManager.Stop()
	a.taskManager.Stop()
	a.localCache.Stop()
	a.systemMonitor.Stop()

	log.Printf("Edge agent %s stopped", a.config.EdgeName)
	return nil
}

// connectionLoop manages WebSocket connection to cloud
func (a *EdgeAgent) connectionLoop() {
	defer a.wg.Done()

	ticker := time.NewTicker(a.config.ReconnectInterval)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			if !a.isConnected() {
				a.connect()
			}
		}
	}
}

// connect establishes WebSocket connection to cloud
func (a *EdgeAgent) connect() {
	log.Printf("Attempting to connect to cloud endpoint: %s", a.config.CloudEndpoint)

	// Setup dialer
	dialer := websocket.DefaultDialer
	if a.config.EnableTLS {
		dialer.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: a.config.SkipTLSVerification,
		}
	}

	// Build URL
	url := a.config.CloudEndpoint + a.config.WebSocketPath
	
	// Connect
	conn, _, err := dialer.Dial(url, nil)
	if err != nil {
		log.Printf("Failed to connect to cloud: %v", err)
		a.setOfflineMode(true)
		return
	}

	a.connMutex.Lock()
	a.conn = conn
	a.connected = true
	a.lastSeen = time.Now()
	a.connMutex.Unlock()

	a.setOfflineMode(false)
	log.Printf("Connected to cloud successfully")

	// Send registration message
	a.sendRegistration()

	// Send any pending offline data
	a.flushOfflineData()

	// Start message reader
	a.wg.Add(1)
	go a.messageReader()
}

// messageReader reads messages from WebSocket
func (a *EdgeAgent) messageReader() {
	defer a.wg.Done()
	defer a.disconnect()

	for {
		select {
		case <-a.ctx.Done():
			return
		default:
			a.connMutex.RLock()
			conn := a.conn
			a.connMutex.RUnlock()

			if conn == nil {
				return
			}

			var message Message
			err := conn.ReadJSON(&message)
			if err != nil {
				log.Printf("Failed to read message: %v", err)
				return
			}

			a.lastSeen = time.Now()
			a.handleMessage(&message)
		}
	}
}

// handleMessage routes messages to appropriate handlers
func (a *EdgeAgent) handleMessage(message *Message) {
	handler, exists := a.messageHandlers[message.Type]
	if !exists {
		log.Printf("No handler for message type: %s", message.Type)
		return
	}

	if err := handler(a, message); err != nil {
		log.Printf("Message handler error for type %s: %v", message.Type, err)
	}
}

// setupMessageHandlers initializes message handlers
func (a *EdgeAgent) setupMessageHandlers() {
	a.messageHandlers["task"] = handleTaskMessage
	a.messageHandlers["config_update"] = handleConfigUpdateMessage
	a.messageHandlers["resource_request"] = handleResourceRequestMessage
	a.messageHandlers["cache_invalidate"] = handleCacheInvalidateMessage
	a.messageHandlers["ping"] = handlePingMessage
	a.messageHandlers["shutdown"] = handleShutdownMessage
}

// sendMessage sends a message to the cloud
func (a *EdgeAgent) sendMessage(msgType string, data map[string]interface{}) error {
	message := Message{
		Type:      msgType,
		ID:        generateUUID(),
		Timestamp: time.Now(),
		Data:      data,
	}

	a.connMutex.RLock()
	conn := a.conn
	connected := a.connected
	a.connMutex.RUnlock()

	if !connected || conn == nil {
		// Store message for offline delivery
		a.storeOfflineMessage(&message)
		return fmt.Errorf("not connected to cloud")
	}

	return conn.WriteJSON(&message)
}

// sendRegistration sends agent registration to cloud
func (a *EdgeAgent) sendRegistration() {
	sysInfo := a.systemMonitor.GetSystemInfo()
	
	registrationData := map[string]interface{}{
		"agent_id":       a.config.AgentID,
		"edge_name":      a.config.EdgeName,
		"region":         a.config.Region,
		"zone":           a.config.Zone,
		"tags":           a.config.Tags,
		"system_info":    sysInfo,
		"capabilities":   a.getCapabilities(),
		"agent_version":  "2.0.0",
		"startup_time":   time.Now(),
	}

	if err := a.sendMessage("register", registrationData); err != nil {
		log.Printf("Failed to send registration: %v", err)
	} else {
		log.Printf("Registration sent successfully")
	}
}

// getCapabilities returns agent capabilities
func (a *EdgeAgent) getCapabilities() []string {
	capabilities := []string{
		"resource_monitoring",
		"task_execution", 
		"local_caching",
		"offline_operation",
		"auto_scaling",
	}

	// Add architecture-specific capabilities
	if runtime.GOARCH == "arm64" || runtime.GOARCH == "arm" {
		capabilities = append(capabilities, "arm_architecture")
	}

	if runtime.GOARCH == "riscv64" {
		capabilities = append(capabilities, "riscv_architecture")
	}

	// Check for GPU availability
	if a.hasGPU() {
		capabilities = append(capabilities, "gpu_acceleration")
	}

	return capabilities
}

// hasGPU checks if GPU is available (simplified)
func (a *EdgeAgent) hasGPU() bool {
	// Simplified GPU detection - in real implementation would check
	// for NVIDIA/AMD/Intel GPU drivers and CUDA/OpenCL support
	return false
}

// heartbeatLoop sends periodic heartbeats
func (a *EdgeAgent) heartbeatLoop() {
	defer a.wg.Done()

	ticker := time.NewTicker(a.config.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.sendHeartbeat()
		}
	}
}

// sendHeartbeat sends heartbeat with basic metrics
func (a *EdgeAgent) sendHeartbeat() {
	metrics := a.systemMonitor.GetCurrentMetrics()
	
	heartbeatData := map[string]interface{}{
		"agent_id":      a.config.AgentID,
		"timestamp":     time.Now(),
		"status":        a.getAgentStatus(),
		"metrics":       metrics,
		"active_tasks":  len(a.pendingTasks),
		"offline_mode":  a.offlineMode,
		"uptime":        time.Since(a.systemMonitor.startTime).Seconds(),
	}

	if err := a.sendMessage("heartbeat", heartbeatData); err != nil {
		// Check if we should enter offline mode
		if time.Since(a.lastSeen) > a.config.OfflineThreshold {
			a.setOfflineMode(true)
		}
	}
}

// metricsLoop collects and sends detailed metrics
func (a *EdgeAgent) metricsLoop() {
	defer a.wg.Done()

	ticker := time.NewTicker(a.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.collectAndSendMetrics()
		}
	}
}

// collectAndSendMetrics collects system metrics and sends to cloud
func (a *EdgeAgent) collectAndSendMetrics() {
	metrics := a.systemMonitor.GetDetailedMetrics()
	
	// Add to local cache
	snapshot := &MetricSnapshot{
		Timestamp: time.Now(),
		Metrics:   metrics,
		AgentID:   a.config.AgentID,
	}
	
	a.metricsCache.Add(snapshot)

	// Send to cloud if connected
	if a.isConnected() {
		metricsData := map[string]interface{}{
			"agent_id": a.config.AgentID,
			"metrics":  metrics,
		}

		if err := a.sendMessage("metrics", metricsData); err != nil {
			log.Printf("Failed to send metrics: %v", err)
			a.storeOfflineMetrics(snapshot)
		}
	} else {
		a.storeOfflineMetrics(snapshot)
	}
}

// taskProcessingLoop processes queued tasks
func (a *EdgeAgent) taskProcessingLoop() {
	defer a.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.processQueuedTasks()
		}
	}
}

// processQueuedTasks processes tasks from the queue
func (a *EdgeAgent) processQueuedTasks() {
	a.taskMutex.Lock()
	tasks := make([]*EdgeTask, 0, len(a.pendingTasks))
	for _, task := range a.pendingTasks {
		if task.Status == TaskStatusQueued {
			tasks = append(tasks, task)
		}
	}
	a.taskMutex.Unlock()

	if len(tasks) == 0 {
		return
	}

	// Sort by priority (higher first)
	// Implementation would sort tasks by priority and deadline

	for _, task := range tasks {
		if a.canExecuteTask(task) {
			a.executeTask(task)
		}
	}
}

// canExecuteTask checks if task can be executed given current resources
func (a *EdgeAgent) canExecuteTask(task *EdgeTask) bool {
	return a.resourceManager.CanAccommodate(task.ResourceReqs)
}

// executeTask executes a task
func (a *EdgeAgent) executeTask(task *EdgeTask) {
	a.taskMutex.Lock()
	task.Status = TaskStatusRunning
	task.StartedAt = time.Now()
	a.taskMutex.Unlock()

	log.Printf("Executing task %s of type %s", task.ID, task.Type)

	// Execute task in background
	go func() {
		defer func() {
			a.resourceManager.Release(task.ResourceReqs)
		}()

		// Reserve resources
		if !a.resourceManager.Reserve(task.ResourceReqs) {
			a.failTask(task, "insufficient resources")
			return
		}

		// Execute based on task type
		result, err := a.taskManager.ExecuteTask(task)
		
		a.taskMutex.Lock()
		task.CompletedAt = time.Now()
		if err != nil {
			task.Status = TaskStatusFailed
			task.Error = err.Error()
		} else {
			task.Status = TaskStatusCompleted
			task.Result = result
		}
		a.taskMutex.Unlock()

		// Send result back to cloud
		a.sendTaskResult(task)

		log.Printf("Task %s completed with status %s", task.ID, task.Status)
	}()
}

// failTask marks a task as failed
func (a *EdgeAgent) failTask(task *EdgeTask, reason string) {
	a.taskMutex.Lock()
	task.Status = TaskStatusFailed
	task.Error = reason
	task.CompletedAt = time.Now()
	a.taskMutex.Unlock()

	a.sendTaskResult(task)
}

// sendTaskResult sends task result to cloud
func (a *EdgeAgent) sendTaskResult(task *EdgeTask) {
	resultData := map[string]interface{}{
		"agent_id": a.config.AgentID,
		"task_id":  task.ID,
		"status":   task.Status,
		"result":   task.Result,
		"error":    task.Error,
		"metrics": map[string]interface{}{
			"execution_time": task.CompletedAt.Sub(task.StartedAt).Seconds(),
			"retry_count":    task.RetryCount,
		},
	}

	if err := a.sendMessage("task_result", resultData); err != nil {
		log.Printf("Failed to send task result for %s: %v", task.ID, err)
		// Store for offline delivery
		a.storeOfflineTaskResult(task)
	}
}

// Utility methods

func (a *EdgeAgent) isConnected() bool {
	a.connMutex.RLock()
	defer a.connMutex.RUnlock()
	return a.connected
}

func (a *EdgeAgent) disconnect() {
	a.connMutex.Lock()
	defer a.connMutex.Unlock()
	
	if a.conn != nil {
		a.conn.Close()
		a.conn = nil
	}
	a.connected = false
	
	log.Printf("Disconnected from cloud")
	a.setOfflineMode(true)
}

func (a *EdgeAgent) setOfflineMode(offline bool) {
	a.offlineMutex.Lock()
	defer a.offlineMutex.Unlock()
	
	if a.offlineMode != offline {
		a.offlineMode = offline
		if offline {
			log.Printf("Entering offline mode")
		} else {
			log.Printf("Exiting offline mode")
		}
	}
}

func (a *EdgeAgent) getAgentStatus() string {
	if a.offlineMode {
		return "offline"
	}
	if a.isConnected() {
		return "connected"
	}
	return "connecting"
}

// Offline data management

func (a *EdgeAgent) storeOfflineMessage(message *Message) {
	// Implementation would store message for later delivery
}

func (a *EdgeAgent) storeOfflineMetrics(snapshot *MetricSnapshot) {
	a.offlineMutex.Lock()
	defer a.offlineMutex.Unlock()
	
	a.offlineMetrics = append(a.offlineMetrics, snapshot)
	
	// Keep only last 1000 metrics
	if len(a.offlineMetrics) > 1000 {
		a.offlineMetrics = a.offlineMetrics[len(a.offlineMetrics)-1000:]
	}
}

func (a *EdgeAgent) storeOfflineTaskResult(task *EdgeTask) {
	a.offlineMutex.Lock()
	defer a.offlineMutex.Unlock()
	
	a.offlineTasks = append(a.offlineTasks, task)
}

func (a *EdgeAgent) flushOfflineData() {
	a.offlineMutex.Lock()
	defer a.offlineMutex.Unlock()
	
	// Send offline metrics
	for _, snapshot := range a.offlineMetrics {
		metricsData := map[string]interface{}{
			"agent_id": snapshot.AgentID,
			"metrics":  snapshot.Metrics,
			"offline":  true,
		}
		a.sendMessage("metrics", metricsData)
	}
	a.offlineMetrics = a.offlineMetrics[:0]

	// Send offline task results
	for _, task := range a.offlineTasks {
		a.sendTaskResult(task)
	}
	a.offlineTasks = a.offlineTasks[:0]
	
	log.Printf("Flushed offline data")
}

// generateUUID generates a simple UUID (implementation would use proper UUID library)
func generateUUID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}