package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
}

type HealthResponse struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Services  map[string]string `json:"services"`
}

type VM struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Status      string    `json:"status"`
	CPU         int       `json:"cpu"`
	Memory      int       `json:"memory"`
	Disk        int       `json:"disk"`
	IP          string    `json:"ip"`
	CreatedAt   time.Time `json:"created_at"`
}

type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

var mockVMs = []VM{
	{
		ID:        "vm-001",
		Name:      "web-server-1",
		Status:    "running",
		CPU:       2,
		Memory:    4096,
		Disk:      50,
		IP:        "10.0.1.10",
		CreatedAt: time.Now().Add(-24 * time.Hour),
	},
	{
		ID:        "vm-002",
		Name:      "database-1",
		Status:    "running",
		CPU:       4,
		Memory:    8192,
		Disk:      100,
		IP:        "10.0.1.11",
		CreatedAt: time.Now().Add(-48 * time.Hour),
	},
	{
		ID:        "vm-003",
		Name:      "cache-server",
		Status:    "stopped",
		CPU:       1,
		Memory:    2048,
		Disk:      20,
		IP:        "10.0.1.12",
		CreatedAt: time.Now().Add(-72 * time.Hour),
	},
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	response := HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now(),
		Services: map[string]string{
			"api":      "online",
			"database": "connected",
			"cache":    "connected",
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func listVMsHandler(w http.ResponseWriter, r *http.Request) {
	response := APIResponse{
		Success: true,
		Data:    mockVMs,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func getVMHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["id"]
	
	for _, vm := range mockVMs {
		if vm.ID == vmID {
			response := APIResponse{
				Success: true,
				Data:    vm,
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}
	}
	
	w.WriteHeader(http.StatusNotFound)
	response := APIResponse{
		Success: false,
		Error:   "VM not found",
	}
	json.NewEncoder(w).Encode(response)
}

func createVMHandler(w http.ResponseWriter, r *http.Request) {
	var newVM VM
	if err := json.NewDecoder(r.Body).Decode(&newVM); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		response := APIResponse{
			Success: false,
			Error:   "Invalid request body",
		}
		json.NewEncoder(w).Encode(response)
		return
	}
	
	newVM.ID = fmt.Sprintf("vm-%03d", len(mockVMs)+1)
	newVM.Status = "creating"
	newVM.CreatedAt = time.Now()
	
	mockVMs = append(mockVMs, newVM)
	
	w.WriteHeader(http.StatusCreated)
	response := APIResponse{
		Success: true,
		Data:    newVM,
	}
	json.NewEncoder(w).Encode(response)
}

func metricsHandler(w http.ResponseWriter, r *http.Request) {
	metrics := map[string]interface{}{
		"total_vms":     len(mockVMs),
		"running_vms":   2,
		"stopped_vms":   1,
		"total_cpu":     7,
		"total_memory":  14336,
		"total_disk":    170,
		"cpu_usage":     45.2,
		"memory_usage":  62.8,
		"disk_usage":    38.5,
	}
	
	response := APIResponse{
		Success: true,
		Data:    metrics,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func websocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	// Send initial connection message
	welcome := map[string]string{
		"type":    "connected",
		"message": "WebSocket connection established",
	}
	conn.WriteJSON(welcome)
	
	// Simple echo server for testing
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}
		
		// Echo the message back
		response := map[string]interface{}{
			"type":      "echo",
			"timestamp": time.Now(),
			"data":      msg,
		}
		
		if err := conn.WriteJSON(response); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

func main() {
	router := mux.NewRouter()
	
	// Health check endpoint
	router.HandleFunc("/health", healthHandler).Methods("GET")
	
	// API endpoints
	api := router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/vms", listVMsHandler).Methods("GET")
	api.HandleFunc("/vms", createVMHandler).Methods("POST")
	api.HandleFunc("/vms/{id}", getVMHandler).Methods("GET")
	api.HandleFunc("/metrics", metricsHandler).Methods("GET")
	
	// WebSocket endpoint
	router.HandleFunc("/ws", websocketHandler)
	
	// Enable CORS for all origins (development only)
	corsHandler := handlers.CORS(
		handlers.AllowedOrigins([]string{"*"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"Content-Type", "Authorization"}),
		handlers.AllowCredentials(),
	)
	
	// Start HTTP server on port 8090
	httpServer := &http.Server{
		Addr:    ":8090",
		Handler: corsHandler(router),
	}
	
	// Start WebSocket server on port 8091 in a goroutine
	go func() {
		wsRouter := mux.NewRouter()
		wsRouter.HandleFunc("/ws", websocketHandler)
		wsRouter.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("WebSocket server is running"))
		})
		
		wsServer := &http.Server{
			Addr:    ":8091",
			Handler: corsHandler(wsRouter),
		}
		
		log.Println("WebSocket server starting on :8091")
		if err := wsServer.ListenAndServe(); err != nil {
			log.Printf("WebSocket server error: %v", err)
		}
	}()
	
	log.Println("API server starting on :8090")
	log.Println("Access the API at http://localhost:8090")
	log.Println("Health check: http://localhost:8090/health")
	log.Println("WebSocket endpoint: ws://localhost:8091/ws")
	
	if err := httpServer.ListenAndServe(); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}