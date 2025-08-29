package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"gopkg.in/yaml.v2"
)

// EdgeAgent represents the lightweight edge computing agent
type EdgeAgent struct {
	config       *Config
	cloudClient  CloudClient
	localCache   *redis.Client
	healthServer *http.Server
	resourceMgr  *ResourceManager
	syncMgr      *SyncManager
	aiInference  *AIInferenceEngine
	metrics      *MetricsCollector
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

// Config holds edge agent configuration
type Config struct {
	// Agent identification
	AgentID     string `yaml:"agent_id"`
	ClusterID   string `yaml:"cluster_id"`
	EdgeRegion  string `yaml:"edge_region"`
	
	// Cloud connectivity
	CloudEndpoint string `yaml:"cloud_endpoint"`
	CloudToken    string `yaml:"cloud_token"`
	SyncInterval  time.Duration `yaml:"sync_interval"`
	
	// Local resources
	LocalRedis    RedisConfig `yaml:"local_redis"`
	ResourceLimits ResourceLimits `yaml:"resource_limits"`
	
	// Edge capabilities
	OfflineMode   bool `yaml:"offline_mode"`
	AIInference   bool `yaml:"ai_inference"`
	LocalDecisions bool `yaml:"local_decisions"`
	
	// Networking
	HealthPort    int    `yaml:"health_port"`
	MetricsPort   int    `yaml:"metrics_port"`
	EdgeNetwork   string `yaml:"edge_network"`
}

type RedisConfig struct {
	Address  string `yaml:"address"`
	Password string `yaml:"password"`
	DB       int    `yaml:"db"`
}

type ResourceLimits struct {
	MaxMemoryMB int64 `yaml:"max_memory_mb"`
	MaxCPUCores int   `yaml:"max_cpu_cores"`
	MaxStorage  int64 `yaml:"max_storage_gb"`
}

// CloudClient handles communication with NovaCron cloud control plane
type CloudClient struct {
	conn        *grpc.ClientConn
	client      interface{} // NovaCronCloudClient when proto is available
	endpoint    string
	token       string
	connected   bool
	lastSync    time.Time
}

// ResourceManager handles edge resource monitoring and management
type ResourceManager struct {
	agent       *EdgeAgent
	cpuUsage    float64
	memoryUsage float64
	diskUsage   float64
	networkIO   NetworkStats
	containers  map[string]*EdgeContainer
	vms         map[string]*EdgeVM
	mutex       sync.RWMutex
}

type NetworkStats struct {
	BytesIn  int64
	BytesOut int64
	PacketsIn int64
	PacketsOut int64
}

type EdgeContainer struct {
	ID          string
	Name        string
	Image       string
	Status      string
	CPUShares   int64
	MemoryMB    int64
	CreatedAt   time.Time
	LastUpdate  time.Time
}

type EdgeVM struct {
	ID         string
	Name       string
	Status     string
	CPUCores   int
	MemoryMB   int64
	DiskGB     int64
	CreatedAt  time.Time
	LastUpdate time.Time
}

// SyncManager handles synchronization with cloud control plane
type SyncManager struct {
	agent        *EdgeAgent
	syncInterval time.Duration
	lastSync     time.Time
	syncErrors   int
	maxOfflineTime time.Duration
	offlineBuffer  []SyncEvent
}

type SyncEvent struct {
	Type      string    `json:"type"`
	Timestamp time.Time `json:"timestamp"`
	Data      interface{} `json:"data"`
	Priority  int       `json:"priority"`
}

// AIInferenceEngine provides local AI inference capabilities
type AIInferenceEngine struct {
	agent      *EdgeAgent
	models     map[string]*AIModel
	enabled    bool
	inferenceQueue chan InferenceRequest
}

type AIModel struct {
	Name        string
	Version     string
	Path        string
	Loaded      bool
	LastUsed    time.Time
	Accuracy    float64
}

type InferenceRequest struct {
	ModelName string
	Input     map[string]interface{}
	Response  chan InferenceResponse
}

type InferenceResponse struct {
	Prediction map[string]interface{}
	Confidence float64
	Latency    time.Duration
	Error      error
}

// MetricsCollector handles edge metrics collection and reporting
type MetricsCollector struct {
	agent     *EdgeAgent
	metrics   map[string]float64
	mutex     sync.RWMutex
	lastReport time.Time
}

// NewEdgeAgent creates a new edge computing agent
func NewEdgeAgent(configPath string) (*EdgeAgent, error) {
	config, err := loadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	agent := &EdgeAgent{
		config: config,
		ctx:    ctx,
		cancel: cancel,
	}
	
	// Initialize local Redis cache
	agent.localCache = redis.NewClient(&redis.Options{
		Addr:     config.LocalRedis.Address,
		Password: config.LocalRedis.Password,
		DB:       config.LocalRedis.DB,
	})
	
	// Test Redis connection
	if err := agent.localCache.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Local Redis not available: %v", err)
	}
	
	// Initialize components
	agent.resourceMgr = NewResourceManager(agent)
	agent.syncMgr = NewSyncManager(agent)
	agent.metrics = NewMetricsCollector(agent)
	
	if config.AIInference {
		agent.aiInference = NewAIInferenceEngine(agent)
	}
	
	// Initialize cloud client
	agent.cloudClient = CloudClient{
		endpoint: config.CloudEndpoint,
		token:    config.CloudToken,
	}
	
	return agent, nil
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		// Return default config if file doesn't exist
		if os.IsNotExist(err) {
			return getDefaultConfig(), nil
		}
		return nil, err
	}
	
	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	
	// Validate and set defaults
	if config.AgentID == "" {
		config.AgentID = fmt.Sprintf("edge-agent-%d", time.Now().Unix())
	}
	
	if config.SyncInterval == 0 {
		config.SyncInterval = 30 * time.Second
	}
	
	if config.HealthPort == 0 {
		config.HealthPort = 8080
	}
	
	return &config, nil
}

func getDefaultConfig() *Config {
	return &Config{
		AgentID:        fmt.Sprintf("edge-agent-%d", time.Now().Unix()),
		ClusterID:      "default",
		EdgeRegion:     "edge-local",
		CloudEndpoint:  "localhost:8090",
		SyncInterval:   30 * time.Second,
		LocalRedis: RedisConfig{
			Address: "localhost:6379",
			DB:      1, // Use different DB than main Redis
		},
		ResourceLimits: ResourceLimits{
			MaxMemoryMB: 1024, // 1GB default limit
			MaxCPUCores: 2,
			MaxStorage:  10, // 10GB
		},
		OfflineMode:    true,
		AIInference:    true,
		LocalDecisions: true,
		HealthPort:     8080,
		MetricsPort:    8081,
		EdgeNetwork:    "edge0",
	}
}

// Start begins edge agent operation
func (a *EdgeAgent) Start() error {
	log.Printf("Starting NovaCron Edge Agent %s", a.config.AgentID)
	
	// Start health check server
	if err := a.startHealthServer(); err != nil {
		return fmt.Errorf("failed to start health server: %w", err)
	}
	
	// Start resource monitoring
	a.wg.Add(1)
	go a.resourceMgr.Start()
	
	// Start metrics collection
	a.wg.Add(1)
	go a.metrics.Start()
	
	// Start sync manager
	a.wg.Add(1)
	go a.syncMgr.Start()
	
	// Start AI inference if enabled
	if a.aiInference != nil {
		a.wg.Add(1)
		go a.aiInference.Start()
	}
	
	// Attempt cloud connection
	go a.connectToCloud()
	
	log.Printf("Edge Agent started successfully")
	log.Printf("Health endpoint: http://localhost:%d/health", a.config.HealthPort)
	log.Printf("Metrics endpoint: http://localhost:%d/metrics", a.config.MetricsPort)
	
	return nil
}

// Stop gracefully shuts down the edge agent
func (a *EdgeAgent) Stop() error {
	log.Printf("Stopping NovaCron Edge Agent %s", a.config.AgentID)
	
	a.cancel()
	
	// Stop health server
	if a.healthServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		a.healthServer.Shutdown(ctx)
	}
	
	// Close cloud connection
	if a.cloudClient.conn != nil {
		a.cloudClient.conn.Close()
	}
	
	// Close Redis connection
	if a.localCache != nil {
		a.localCache.Close()
	}
	
	// Wait for all goroutines to finish
	a.wg.Wait()
	
	log.Printf("Edge Agent stopped successfully")
	return nil
}

func (a *EdgeAgent) startHealthServer() error {
	router := mux.NewRouter()
	
	// Health check endpoint
	router.HandleFunc("/health", a.handleHealth).Methods("GET")
	router.HandleFunc("/status", a.handleStatus).Methods("GET")
	router.HandleFunc("/resources", a.handleResources).Methods("GET")
	router.HandleFunc("/sync", a.handleSync).Methods("POST")
	
	a.healthServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", a.config.HealthPort),
		Handler: router,
	}
	
	go func() {
		if err := a.healthServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Health server error: %v", err)
		}
	}()
	
	return nil
}

func (a *EdgeAgent) connectToCloud() {
	for {
		select {
		case <-a.ctx.Done():
			return
		default:
			if err := a.establishCloudConnection(); err != nil {
				log.Printf("Failed to connect to cloud: %v", err)
				time.Sleep(10 * time.Second)
				continue
			}
			
			log.Printf("Connected to cloud control plane")
			a.cloudClient.connected = true
			return
		}
	}
}

func (a *EdgeAgent) establishCloudConnection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	conn, err := grpc.DialContext(ctx, a.config.CloudEndpoint, 
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock())
	if err != nil {
		return err
	}
	
	a.cloudClient.conn = conn
	// Initialize proto client here when available
	
	return nil
}

// HTTP Handlers
func (a *EdgeAgent) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":     "healthy",
		"agent_id":   a.config.AgentID,
		"cluster_id": a.config.ClusterID,
		"uptime":     time.Since(time.Now()).String(), // This should track actual uptime
		"cloud_connected": a.cloudClient.connected,
		"local_cache": a.localCache.Ping(a.ctx).Err() == nil,
		"timestamp": time.Now().Unix(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (a *EdgeAgent) handleStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"agent": map[string]interface{}{
			"id":      a.config.AgentID,
			"cluster": a.config.ClusterID,
			"region":  a.config.EdgeRegion,
		},
		"resources": a.resourceMgr.GetResourceSummary(),
		"sync": map[string]interface{}{
			"last_sync":    a.syncMgr.lastSync,
			"sync_errors":  a.syncMgr.syncErrors,
			"offline_mode": a.config.OfflineMode,
		},
		"ai": map[string]interface{}{
			"enabled": a.config.AIInference,
			"models":  len(a.aiInference.models),
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (a *EdgeAgent) handleResources(w http.ResponseWriter, r *http.Request) {
	resources := a.resourceMgr.GetDetailedResources()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resources)
}

func (a *EdgeAgent) handleSync(w http.ResponseWriter, r *http.Request) {
	// Trigger immediate sync with cloud
	go a.syncMgr.TriggerSync()
	
	response := map[string]string{
		"message": "Sync triggered",
		"agent_id": a.config.AgentID,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Component initialization functions
func NewResourceManager(agent *EdgeAgent) *ResourceManager {
	return &ResourceManager{
		agent:      agent,
		containers: make(map[string]*EdgeContainer),
		vms:        make(map[string]*EdgeVM),
	}
}

func NewSyncManager(agent *EdgeAgent) *SyncManager {
	return &SyncManager{
		agent:          agent,
		syncInterval:   agent.config.SyncInterval,
		maxOfflineTime: 72 * time.Hour, // 72 hours offline capability
		offlineBuffer:  make([]SyncEvent, 0),
	}
}

func NewAIInferenceEngine(agent *EdgeAgent) *AIInferenceEngine {
	return &AIInferenceEngine{
		agent:          agent,
		models:         make(map[string]*AIModel),
		enabled:        true,
		inferenceQueue: make(chan InferenceRequest, 100),
	}
}

func NewMetricsCollector(agent *EdgeAgent) *MetricsCollector {
	return &MetricsCollector{
		agent:   agent,
		metrics: make(map[string]float64),
	}
}

// Component start methods
func (rm *ResourceManager) Start() {
	defer rm.agent.wg.Done()
	
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-rm.agent.ctx.Done():
			return
		case <-ticker.C:
			rm.collectResourceMetrics()
		}
	}
}

func (sm *SyncManager) Start() {
	defer sm.agent.wg.Done()
	
	ticker := time.NewTicker(sm.syncInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-sm.agent.ctx.Done():
			return
		case <-ticker.C:
			sm.performSync()
		}
	}
}

func (ai *AIInferenceEngine) Start() {
	defer ai.agent.wg.Done()
	
	// Load AI models
	ai.loadModels()
	
	// Process inference requests
	for {
		select {
		case <-ai.agent.ctx.Done():
			return
		case req := <-ai.inferenceQueue:
			ai.processInference(req)
		}
	}
}

func (mc *MetricsCollector) Start() {
	defer mc.agent.wg.Done()
	
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-mc.agent.ctx.Done():
			return
		case <-ticker.C:
			mc.collectMetrics()
		}
	}
}

// Implementation stubs for component methods
func (rm *ResourceManager) collectResourceMetrics() {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()
	
	// Collect CPU, memory, disk metrics
	// Implementation would use system calls or libraries like gopsutil
	rm.cpuUsage = 45.0  // Placeholder
	rm.memoryUsage = 60.0
	rm.diskUsage = 25.0
}

func (rm *ResourceManager) GetResourceSummary() map[string]interface{} {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	
	return map[string]interface{}{
		"cpu_usage":    rm.cpuUsage,
		"memory_usage": rm.memoryUsage,
		"disk_usage":   rm.diskUsage,
		"containers":   len(rm.containers),
		"vms":          len(rm.vms),
	}
}

func (rm *ResourceManager) GetDetailedResources() map[string]interface{} {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	
	return map[string]interface{}{
		"summary":    rm.GetResourceSummary(),
		"containers": rm.containers,
		"vms":        rm.vms,
		"network":    rm.networkIO,
	}
}

func (sm *SyncManager) performSync() {
	if !sm.agent.cloudClient.connected {
		log.Printf("Cloud not connected, skipping sync")
		return
	}
	
	// Perform sync with cloud control plane
	sm.lastSync = time.Now()
	log.Printf("Sync completed at %v", sm.lastSync)
}

func (sm *SyncManager) TriggerSync() {
	sm.performSync()
}

func (ai *AIInferenceEngine) loadModels() {
	// Load AI models for edge inference
	log.Printf("Loading AI models for edge inference")
}

func (ai *AIInferenceEngine) processInference(req InferenceRequest) {
	// Process AI inference request
	start := time.Now()
	
	response := InferenceResponse{
		Prediction: map[string]interface{}{
			"result": "placeholder_result",
		},
		Confidence: 0.85,
		Latency:    time.Since(start),
	}
	
	req.Response <- response
}

func (mc *MetricsCollector) collectMetrics() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()
	
	// Collect various metrics
	mc.metrics["edge_agent_uptime"] = time.Since(time.Now()).Seconds()
	mc.metrics["edge_agent_memory_mb"] = 45.0 // Placeholder
	mc.metrics["edge_decisions_per_sec"] = 12.5
	
	mc.lastReport = time.Now()
}

// Main function
func main() {
	configPath := os.Getenv("EDGE_AGENT_CONFIG")
	if configPath == "" {
		configPath = "/etc/novacron/edge-agent.yaml"
	}
	
	agent, err := NewEdgeAgent(configPath)
	if err != nil {
		log.Fatalf("Failed to create edge agent: %v", err)
	}
	
	if err := agent.Start(); err != nil {
		log.Fatalf("Failed to start edge agent: %v", err)
	}
	
	// Wait for termination signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	<-sigChan
	log.Printf("Received termination signal")
	
	if err := agent.Stop(); err != nil {
		log.Printf("Error stopping agent: %v", err)
	}
}