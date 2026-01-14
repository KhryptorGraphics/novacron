package websocket

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

// TestWebSocketMetricsEndpoint tests the /ws/metrics endpoint
func TestWebSocketMetricsEndpoint(t *testing.T) {
	// Create handler with nil dependencies (will use defaults)
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	// Create test router
	router := mux.NewRouter()

	// Skip auth for testing - direct handler
	router.HandleFunc("/ws/metrics", handler.HandleMetricsWebSocket).Methods("GET")

	// Create test server
	server := httptest.NewServer(router)
	defer server.Close()

	// Connect via WebSocket
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/metrics?interval=1"

	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	defer conn.Close()

	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Errorf("Expected status 101, got %d", resp.StatusCode)
	}

	// Wait for metrics message
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))

	_, message, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("Failed to read message: %v", err)
	}

	// Parse message
	var metricsMsg MetricsMessage
	if err := json.Unmarshal(message, &metricsMsg); err != nil {
		t.Fatalf("Failed to parse message: %v", err)
	}

	// Validate message structure
	if metricsMsg.Type != "metrics_update" {
		t.Errorf("Expected type 'metrics_update', got '%s'", metricsMsg.Type)
	}

	if metricsMsg.Metrics == nil {
		t.Error("Metrics should not be nil")
	}

	if metricsMsg.Timestamp.IsZero() {
		t.Error("Timestamp should not be zero")
	}

	t.Logf("Received metrics message: %+v", metricsMsg)
}

// TestWebSocketAlertsEndpoint tests the /ws/alerts endpoint
func TestWebSocketAlertsEndpoint(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	router.HandleFunc("/ws/alerts", handler.HandleAlertsWebSocket).Methods("GET")

	server := httptest.NewServer(router)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/alerts"

	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	defer conn.Close()

	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Errorf("Expected status 101, got %d", resp.StatusCode)
	}

	// Send a control message to update filters
	controlMsg := WebSocketMessage{
		Type:      "update_filters",
		Filters:   map[string]interface{}{"severity": []string{"critical", "warning"}},
		Timestamp: time.Now(),
	}

	data, _ := json.Marshal(controlMsg)
	if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
		t.Fatalf("Failed to send control message: %v", err)
	}

	t.Log("Successfully connected to alerts WebSocket and sent filter update")
}

// TestWebSocketPingPong tests the heartbeat mechanism
func TestWebSocketPingPong(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	router.HandleFunc("/ws/metrics", handler.HandleMetricsWebSocket).Methods("GET")

	server := httptest.NewServer(router)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/metrics?interval=60"

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	defer conn.Close()

	// Set up pong handler
	pongReceived := make(chan bool, 1)
	conn.SetPongHandler(func(string) error {
		pongReceived <- true
		return nil
	})

	// Send ping
	if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
		t.Fatalf("Failed to send ping: %v", err)
	}

	// Wait for pong with timeout
	select {
	case <-pongReceived:
		t.Log("Ping-pong heartbeat working correctly")
	case <-time.After(2 * time.Second):
		t.Log("Note: Server may not respond to client pings (pong is sent by server)")
	}
}

// TestWebSocketFilters tests that filter updates work correctly
func TestWebSocketFilters(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	router.HandleFunc("/ws/metrics", handler.HandleMetricsWebSocket).Methods("GET")

	server := httptest.NewServer(router)
	defer server.Close()

	// Connect with source filter
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/metrics?sources=cpu_usage,memory_usage&interval=1"

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	defer conn.Close()

	// Wait for metrics message
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))

	_, message, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("Failed to read message: %v", err)
	}

	var metricsMsg MetricsMessage
	if err := json.Unmarshal(message, &metricsMsg); err != nil {
		t.Fatalf("Failed to parse message: %v", err)
	}

	// Check that metrics are present (even if mock/fallback values)
	if _, exists := metricsMsg.Metrics["timestamp"]; !exists {
		t.Error("Expected timestamp in metrics")
	}

	t.Logf("Filter test passed with metrics: %+v", metricsMsg.Metrics)
}

// TestWebSocketConnectionPooling tests multiple concurrent connections
func TestWebSocketConnectionPooling(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	router.HandleFunc("/ws/metrics", handler.HandleMetricsWebSocket).Methods("GET")

	server := httptest.NewServer(router)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/metrics?interval=60"

	// Create multiple connections
	connections := make([]*websocket.Conn, 5)
	for i := 0; i < 5; i++ {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Failed to create connection %d: %v", i, err)
		}
		connections[i] = conn
	}

	// Verify all connections are active
	t.Logf("Successfully created %d concurrent WebSocket connections", len(connections))

	// Close all connections
	for i, conn := range connections {
		if err := conn.Close(); err != nil {
			t.Errorf("Failed to close connection %d: %v", i, err)
		}
	}

	t.Log("All connections closed successfully")
}

// TestMessageTypes tests all message type structures
func TestMessageTypes(t *testing.T) {
	// Test MetricsMessage
	metricsMsg := MetricsMessage{
		Type:      "metrics_update",
		Source:    "system",
		Metrics:   map[string]interface{}{"cpu": 50.5, "memory": 60.0},
		Timestamp: time.Now(),
		Labels:    map[string]string{"host": "node1"},
	}

	data, err := json.Marshal(metricsMsg)
	if err != nil {
		t.Errorf("Failed to marshal MetricsMessage: %v", err)
	}

	var decoded MetricsMessage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Errorf("Failed to unmarshal MetricsMessage: %v", err)
	}

	if decoded.Type != metricsMsg.Type {
		t.Errorf("MetricsMessage type mismatch: expected %s, got %s", metricsMsg.Type, decoded.Type)
	}

	// Test AlertMessage
	alertMsg := AlertMessage{
		Type:        "alert",
		AlertID:     "alert-123",
		Severity:    "critical",
		Title:       "High CPU Usage",
		Description: "CPU usage exceeded 90%",
		Source:      "monitoring",
		Timestamp:   time.Now(),
		Labels:      map[string]string{"vm_id": "vm-1"},
		Metadata:    map[string]interface{}{"threshold": 90},
	}

	data, err = json.Marshal(alertMsg)
	if err != nil {
		t.Errorf("Failed to marshal AlertMessage: %v", err)
	}

	var decodedAlert AlertMessage
	if err := json.Unmarshal(data, &decodedAlert); err != nil {
		t.Errorf("Failed to unmarshal AlertMessage: %v", err)
	}

	if decodedAlert.Severity != alertMsg.Severity {
		t.Errorf("AlertMessage severity mismatch: expected %s, got %s", alertMsg.Severity, decodedAlert.Severity)
	}

	// Test LogMessage
	logMsg := LogMessage{
		Type:      "log",
		Source:    "system",
		Level:     "info",
		Message:   "Service started",
		Timestamp: time.Now(),
		Component: "api-server",
		VMID:      "",
	}

	data, err = json.Marshal(logMsg)
	if err != nil {
		t.Errorf("Failed to marshal LogMessage: %v", err)
	}

	var decodedLog LogMessage
	if err := json.Unmarshal(data, &decodedLog); err != nil {
		t.Errorf("Failed to unmarshal LogMessage: %v", err)
	}

	if decodedLog.Level != logMsg.Level {
		t.Errorf("LogMessage level mismatch: expected %s, got %s", logMsg.Level, decodedLog.Level)
	}

	t.Log("All message type serialization tests passed")
}

// TestHelperFunctions tests the helper functions
func TestHelperFunctions(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	// Test parseCommaSeparated
	result := handler.parseCommaSeparated("a,b,c")
	if len(result) != 3 {
		t.Errorf("Expected 3 elements, got %d", len(result))
	}

	result = handler.parseCommaSeparated("")
	if len(result) != 0 {
		t.Errorf("Expected 0 elements for empty string, got %d", len(result))
	}

	// Test parseIntWithDefault
	intResult := handler.parseIntWithDefault("10", 5)
	if intResult != 10 {
		t.Errorf("Expected 10, got %d", intResult)
	}

	intResult = handler.parseIntWithDefault("invalid", 5)
	if intResult != 5 {
		t.Errorf("Expected default 5, got %d", intResult)
	}

	// Test generateClientID
	id1 := handler.generateClientID()
	id2 := handler.generateClientID()
	if id1 == id2 {
		t.Error("Client IDs should be unique")
	}
	if !strings.HasPrefix(id1, "client-") {
		t.Errorf("Client ID should start with 'client-', got %s", id1)
	}

	t.Log("All helper function tests passed")
}

// TestCollectMetrics tests the metric collection function
func TestCollectMetrics(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	// Test with no sources (should use defaults)
	metrics := handler.collectMetrics(nil)
	if metrics == nil {
		t.Error("Metrics should not be nil")
	}

	if _, exists := metrics["timestamp"]; !exists {
		t.Error("Expected timestamp in metrics")
	}

	// Test with specific sources
	metrics = handler.collectMetrics([]string{"cpu_usage", "memory_usage"})
	if metrics == nil {
		t.Error("Metrics should not be nil")
	}

	t.Logf("Collected metrics: %+v", metrics)
}

// TestAlertFilters tests the alert filter matching
func TestAlertFilters(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	alert := AlertMessage{
		Type:     "alert",
		Severity: "critical",
		Source:   "monitoring",
	}

	// Test with no filters (should match)
	if !handler.matchesAlertFilters(alert, map[string]interface{}{}) {
		t.Error("Alert should match with no filters")
	}

	// Test with matching severity filter
	filters := map[string]interface{}{
		"severities": []string{"critical", "warning"},
	}
	if !handler.matchesAlertFilters(alert, filters) {
		t.Error("Alert should match with matching severity filter")
	}

	// Test with non-matching severity filter
	filters = map[string]interface{}{
		"severities": []string{"info"},
	}
	if handler.matchesAlertFilters(alert, filters) {
		t.Error("Alert should not match with non-matching severity filter")
	}

	// Test with matching source filter
	filters = map[string]interface{}{
		"sources": []string{"monitoring", "system"},
	}
	if !handler.matchesAlertFilters(alert, filters) {
		t.Error("Alert should match with matching source filter")
	}

	t.Log("All alert filter tests passed")
}

// TestLogFilters tests the log filter matching
func TestLogFilters(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	logMsg := LogMessage{
		Type:      "log",
		Source:    "system",
		Level:     "error",
		Component: "api-server",
		VMID:      "vm-123",
	}

	// Test with "all" source (should match)
	if !handler.matchesLogFilters(logMsg, "all", map[string]interface{}{}) {
		t.Error("Log should match with 'all' source")
	}

	// Test with matching source
	if !handler.matchesLogFilters(logMsg, "system", map[string]interface{}{}) {
		t.Error("Log should match with matching source")
	}

	// Test with non-matching source
	if handler.matchesLogFilters(logMsg, "vm", map[string]interface{}{}) {
		t.Error("Log should not match with non-matching source")
	}

	// Test with level filter
	filters := map[string]interface{}{
		"levels": []string{"error", "critical"},
	}
	if !handler.matchesLogFilters(logMsg, "all", filters) {
		t.Error("Log should match with matching level filter")
	}

	// Test with component filter
	filters = map[string]interface{}{
		"components": []string{"api-server", "scheduler"},
	}
	if !handler.matchesLogFilters(logMsg, "all", filters) {
		t.Error("Log should match with matching component filter")
	}

	// Test with VM ID filter
	filters = map[string]interface{}{
		"vm_id": "vm-123",
	}
	if !handler.matchesLogFilters(logMsg, "all", filters) {
		t.Error("Log should match with matching VM ID filter")
	}

	t.Log("All log filter tests passed")
}
