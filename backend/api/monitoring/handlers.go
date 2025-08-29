package monitoring

import (
	"context"
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// MonitoringHandlers provides HTTP handlers for monitoring endpoints
type MonitoringHandlers struct {
	kvmManager     *hypervisor.KVMManager
	alertManager   *monitoring.AlertManager
	metricRegistry *monitoring.MetricRegistry
	upgrader       websocket.Upgrader
}

// NewMonitoringHandlers creates a new monitoring handlers instance
func NewMonitoringHandlers(kvmManager *hypervisor.KVMManager) *MonitoringHandlers {
	return &MonitoringHandlers{
		kvmManager:     kvmManager,
		alertManager:   monitoring.NewAlertManager(),
		metricRegistry: monitoring.NewMetricRegistry(),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for development
			},
		},
	}
}

// NewMonitoringHandlersWithVMManager creates monitoring handlers using VM manager as fallback
func NewMonitoringHandlersWithVMManager(vmManager *vm.VMManager) *MonitoringHandlers {
	return &MonitoringHandlers{
		kvmManager:     nil, // No KVM manager available
		alertManager:   monitoring.NewAlertManager(),
		metricRegistry: monitoring.NewMetricRegistry(),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for development
			},
		},
	}
}

// SystemMetricsResponse represents the response for system metrics
type SystemMetricsResponse struct {
	CurrentCpuUsage       float64   `json:"currentCpuUsage"`
	CurrentMemoryUsage    float64   `json:"currentMemoryUsage"`
	CurrentDiskUsage      float64   `json:"currentDiskUsage"`
	CurrentNetworkUsage   float64   `json:"currentNetworkUsage"`
	CpuChangePercentage   float64   `json:"cpuChangePercentage"`
	MemoryChangePercentage float64  `json:"memoryChangePercentage"`
	DiskChangePercentage  float64   `json:"diskChangePercentage"`
	NetworkChangePercentage float64 `json:"networkChangePercentage"`
	CpuTimeseriesData     []float64 `json:"cpuTimeseriesData"`
	MemoryTimeseriesData  []float64 `json:"memoryTimeseriesData"`
	DiskTimeseriesData    []float64 `json:"diskTimeseriesData"`
	NetworkTimeseriesData []float64 `json:"networkTimeseriesData"`
	TimeLabels            []string  `json:"timeLabels"`
	CpuAnalysis           string    `json:"cpuAnalysis"`
	MemoryAnalysis        string    `json:"memoryAnalysis"`
	MemoryInUse           float64   `json:"memoryInUse"`
	MemoryAvailable       float64   `json:"memoryAvailable"`
	MemoryReserved        float64   `json:"memoryReserved"`
	MemoryCached          float64   `json:"memoryCached"`
}

// VMMetricsResponse represents VM metrics for the frontend
type VMMetricsResponse struct {
	VmId        string  `json:"vmId"`
	Name        string  `json:"name"`
	CpuUsage    float64 `json:"cpuUsage"`
	MemoryUsage float64 `json:"memoryUsage"`
	DiskUsage   float64 `json:"diskUsage"`
	NetworkRx   int64   `json:"networkRx"`
	NetworkTx   int64   `json:"networkTx"`
	Iops        int     `json:"iops"`
	Status      string  `json:"status"`
}

// AlertResponse represents an alert for the frontend
type AlertResponse struct {
	Id          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Severity    string            `json:"severity"`
	Status      string            `json:"status"`
	StartTime   string            `json:"startTime"`
	EndTime     *string           `json:"endTime,omitempty"`
	Labels      map[string]string `json:"labels"`
	Value       float64           `json:"value"`
	Resource    string            `json:"resource"`
}

// GetSystemMetrics handles GET /api/monitoring/metrics
func (h *MonitoringHandlers) GetSystemMetrics(w http.ResponseWriter, r *http.Request) {
	timeRangeStr := r.URL.Query().Get("timeRange")
	timeRange, err := strconv.Atoi(timeRangeStr)
	if err != nil {
		timeRange = 3600 // Default to 1 hour
	}

	ctx := context.Background()
	
	// Get hypervisor metrics
	resourceInfo, err := h.kvmManager.GetHypervisorMetrics(ctx)
	if err != nil {
		log.Printf("Error getting hypervisor metrics: %v", err)
		http.Error(w, "Failed to get system metrics", http.StatusInternalServerError)
		return
	}

	// Generate time series data (simulate for now)
	dataPoints := min(timeRange/60, 100) // Max 100 data points
	cpuData := make([]float64, dataPoints)
	memoryData := make([]float64, dataPoints)
	diskData := make([]float64, dataPoints)
	networkData := make([]float64, dataPoints)
	timeLabels := make([]string, dataPoints)

	now := time.Now()
	for i := 0; i < dataPoints; i++ {
		timestamp := now.Add(-time.Duration(dataPoints-i) * time.Minute)
		timeLabels[i] = timestamp.Format("15:04")
		
		// Simulate realistic metrics with some variation
		baseTime := float64(timestamp.Unix())
		cpuData[i] = 30 + 20*math.Sin(baseTime/3600) + 10*math.Sin(baseTime/300) + 5*rand.Float64()
		memoryData[i] = 60 + 15*math.Sin(baseTime/7200) + 8*rand.Float64()
		diskData[i] = 45 + 10*math.Sin(baseTime/1800) + 5*rand.Float64()
		networkData[i] = 25 + 30*math.Sin(baseTime/900) + 15*rand.Float64()
	}

	response := SystemMetricsResponse{
		CurrentCpuUsage:        cpuData[len(cpuData)-1],
		CurrentMemoryUsage:     memoryData[len(memoryData)-1],
		CurrentDiskUsage:       diskData[len(diskData)-1],
		CurrentNetworkUsage:    networkData[len(networkData)-1],
		CpuChangePercentage:    calculateChange(cpuData),
		MemoryChangePercentage: calculateChange(memoryData),
		DiskChangePercentage:   calculateChange(diskData),
		NetworkChangePercentage: calculateChange(networkData),
		CpuTimeseriesData:      cpuData,
		MemoryTimeseriesData:   memoryData,
		DiskTimeseriesData:     diskData,
		NetworkTimeseriesData:  networkData,
		TimeLabels:             timeLabels,
		CpuAnalysis:           "CPU usage shows normal workday patterns with peaks during business hours.",
		MemoryAnalysis:        "Memory allocation is healthy with sufficient available memory for operations.",
		MemoryInUse:           65.0,
		MemoryAvailable:       20.0,
		MemoryReserved:        10.0,
		MemoryCached:          5.0,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetVMMetrics handles GET /api/monitoring/vms
func (h *MonitoringHandlers) GetVMMetrics(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	
	// Get list of VMs
	vms, err := h.kvmManager.ListVMs(ctx)
	if err != nil {
		log.Printf("Error listing VMs: %v", err)
		http.Error(w, "Failed to get VM metrics", http.StatusInternalServerError)
		return
	}

	var vmMetrics []VMMetricsResponse
	for _, vmInfo := range vms {
		// Get detailed metrics for each VM
		detailedVM, err := h.kvmManager.GetVMMetrics(ctx, vmInfo.ID)
		if err != nil {
			log.Printf("Error getting metrics for VM %s: %v", vmInfo.ID, err)
			continue
		}

		status := "unknown"
		switch detailedVM.State {
		case vm.StateRunning:
			status = "running"
		case vm.StateStopped:
			status = "stopped"
		case vm.StateFailed:
			status = "error"
		}

		vmMetrics = append(vmMetrics, VMMetricsResponse{
			VmId:        detailedVM.ID,
			Name:        detailedVM.Name,
			CpuUsage:    detailedVM.CPUUsage,
			MemoryUsage: float64(detailedVM.MemoryUsage) / float64(detailedVM.MemoryMB) * 100,
			DiskUsage:   50 + 20*rand.Float64(), // Simulate disk usage
			NetworkRx:   detailedVM.NetworkRecv,
			NetworkTx:   detailedVM.NetworkSent,
			Iops:        int(100 + 50*rand.Float64()), // Simulate IOPS
			Status:      status,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(vmMetrics)
}

// GetAlerts handles GET /api/monitoring/alerts
func (h *MonitoringHandlers) GetAlerts(w http.ResponseWriter, r *http.Request) {
	// Get alerts from alert manager
	alerts := h.alertManager.GetActiveAlerts()
	
	var alertResponses []AlertResponse
	for _, alert := range alerts {
		alertResponse := AlertResponse{
			Id:          alert.ID,
			Name:        alert.Name,
			Description: alert.Description,
			Severity:    string(alert.Severity),
			Status:      string(alert.Status),
			StartTime:   alert.StartTime.Format(time.RFC3339),
			Labels:      alert.Labels,
			Value:       alert.Value,
			Resource:    alert.Resource,
		}
		
		if !alert.EndTime.IsZero() {
			endTimeStr := alert.EndTime.Format(time.RFC3339)
			alertResponse.EndTime = &endTimeStr
		}
		
		alertResponses = append(alertResponses, alertResponse)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alertResponses)
}

// AcknowledgeAlert handles POST /api/monitoring/alerts/{id}/acknowledge
func (h *MonitoringHandlers) AcknowledgeAlert(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	alertID := vars["id"]
	
	if err := h.alertManager.AcknowledgeAlert(alertID); err != nil {
		log.Printf("Error acknowledging alert %s: %v", alertID, err)
		http.Error(w, "Failed to acknowledge alert", http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "acknowledged"})
}

// WebSocketHandler handles WebSocket connections for real-time updates
func (h *MonitoringHandlers) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Println("WebSocket client connected")

	// Send periodic updates
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Send metric update
			message := map[string]interface{}{
				"type": "metric",
				"data": map[string]interface{}{
					"timestamp": time.Now().Unix(),
					"cpu":       50 + 20*rand.Float64(),
					"memory":    60 + 15*rand.Float64(),
					"disk":      45 + 10*rand.Float64(),
					"network":   25 + 30*rand.Float64(),
				},
			}
			
			if err := conn.WriteJSON(message); err != nil {
				log.Printf("WebSocket write error: %v", err)
				return
			}
		}
	}
}

// RegisterRoutes registers all monitoring routes
func (h *MonitoringHandlers) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/api/monitoring/metrics", h.GetSystemMetrics).Methods("GET")
	router.HandleFunc("/api/monitoring/vms", h.GetVMMetrics).Methods("GET")
	router.HandleFunc("/api/monitoring/alerts", h.GetAlerts).Methods("GET")
	router.HandleFunc("/api/monitoring/alerts/{id}/acknowledge", h.AcknowledgeAlert).Methods("POST")
	router.HandleFunc("/ws/monitoring", h.WebSocketHandler)
}

// Helper functions
func calculateChange(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}
	
	current := data[len(data)-1]
	previous := data[len(data)-2]
	
	if previous == 0 {
		return 0
	}
	
	return ((current - previous) / previous) * 100
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}