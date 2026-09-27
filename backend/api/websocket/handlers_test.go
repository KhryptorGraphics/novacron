package websocket

import (
	"context"
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

func passthroughRoleGuard(_ string, next http.HandlerFunc) http.Handler {
	return next
}

func TestRegisterWebSocketRoutesIncludesCanonicalAliases(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, passthroughRoleGuard)

	server := httptest.NewServer(router)
	defer server.Close()

	for _, path := range []string{"/ws/metrics?interval=1", "/api/ws/metrics?interval=1"} {
		wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + path
		conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("failed to connect to %s: %v", path, err)
		}

		if resp.StatusCode != http.StatusSwitchingProtocols {
			conn.Close()
			t.Fatalf("expected status 101 from %s, got %d", path, resp.StatusCode)
		}

		if err := conn.Close(); err != nil {
			t.Fatalf("close websocket %s: %v", path, err)
		}
	}
}

// TestWebSocketMetricsEndpoint tests the /ws/metrics endpoint
func TestWebSocketMetricsEndpoint(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, MetricsInterval(20*time.Millisecond), logger)
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
	if metricsMsg.Type != "metric" {
		t.Errorf("Expected type 'metric', got '%s'", metricsMsg.Type)
	}

	if metricsMsg.Metrics == nil {
		t.Error("Metrics should not be nil")
	}

	if metricsMsg.Timestamp.IsZero() {
		t.Error("Timestamp should not be zero")
	}

	t.Logf("Received metrics message: %+v", metricsMsg)
}

func TestRegisterWebSocketRoutesSupportsCanonicalMetricsPrefix(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, MetricsInterval(20*time.Millisecond), logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, func(_ string, next http.HandlerFunc) http.Handler {
		return http.HandlerFunc(next)
	})

	server := httptest.NewServer(router)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/api/ws/metrics?interval=1"

	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to canonical metrics WebSocket: %v", err)
	}
	defer conn.Close()

	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Errorf("Expected status 101, got %d", resp.StatusCode)
	}

	conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	_, message, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("Failed to read canonical metrics message: %v", err)
	}

	var metricsMsg MetricsMessage
	if err := json.Unmarshal(message, &metricsMsg); err != nil {
		t.Fatalf("Failed to parse canonical metrics message: %v", err)
	}

	if metricsMsg.Type != "metric" {
		t.Errorf("Expected type 'metric', got '%s'", metricsMsg.Type)
	}
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

// TestWebSocketMetricsDelivery replaces the old "filters" test: source
// filtering on the metrics channel was never real (collectMetrics ignored its
// sources param), and the spec-mandated single shared sampler makes per-
// client polling intervals vestigial too — this now just proves a connected
// metrics client receives the shared sample.
func TestWebSocketMetricsDelivery(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, MetricsInterval(20*time.Millisecond), logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	router.HandleFunc("/ws/metrics", handler.HandleMetricsWebSocket).Methods("GET")

	server := httptest.NewServer(router)
	defer server.Close()

	// sources/interval query params are parsed for URL back-compat but no
	// longer drive delivery (documented behavior change).
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/metrics?sources=cpu_usage,memory_usage&interval=1"

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket: %v", err)
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(3 * time.Second))

	_, message, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("Failed to read message: %v", err)
	}

	var metricsMsg MetricsMessage
	if err := json.Unmarshal(message, &metricsMsg); err != nil {
		t.Fatalf("Failed to parse message: %v", err)
	}

	if _, exists := metricsMsg.Metrics["timestamp"]; !exists {
		t.Error("Expected timestamp in metrics")
	}

	t.Logf("Metrics delivery test passed with metrics: %+v", metricsMsg.Metrics)
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
		Type:      "metric",
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

	// Test AlertMessage ({type, data, timestamp} shape)
	alertMsg := AlertMessage{
		Type: "security_alert",
		Data: map[string]interface{}{
			"severity":    "critical",
			"title":       "High CPU Usage",
			"description": "CPU usage exceeded 90%",
			"source":      "monitoring",
			"threshold":   90,
		},
		Timestamp: time.Now(),
	}

	data, err = json.Marshal(alertMsg)
	if err != nil {
		t.Errorf("Failed to marshal AlertMessage: %v", err)
	}

	var decodedAlert AlertMessage
	if err := json.Unmarshal(data, &decodedAlert); err != nil {
		t.Errorf("Failed to unmarshal AlertMessage: %v", err)
	}

	if decodedAlert.Data["severity"] != alertMsg.Data["severity"] {
		t.Errorf("AlertMessage severity mismatch: expected %v, got %v", alertMsg.Data["severity"], decodedAlert.Data["severity"])
	}
	if decodedAlert.Type != alertMsg.Type {
		t.Errorf("AlertMessage type mismatch: expected %s, got %s", alertMsg.Type, decodedAlert.Type)
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

// TestSampleMetricsNoDataWhenProviderNil replaces TestCollectMetrics: with no
// MetricsProvider dep configured, sampleMetrics reports status "no_data" and
// never fabricates zero-valued numeric fields.
func TestSampleMetricsNoDataWhenProviderNil(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	metrics := handler.sampleMetrics()
	if metrics == nil {
		t.Fatal("sampleMetrics should not return nil")
	}
	if metrics["status"] != "no_data" {
		t.Errorf("expected status 'no_data', got %v", metrics["status"])
	}
	for _, key := range []string{"cpu_usage", "memory_usage", "disk_usage", "network_io"} {
		if _, exists := metrics[key]; exists {
			t.Errorf("no_data sample should not fabricate %q", key)
		}
	}
	if _, exists := metrics["timestamp"]; !exists {
		t.Error("expected timestamp in no_data sample")
	}
}

// TestSampleMetricsUsesInjectedProvider proves sampleMetrics returns the
// injected provider's sample verbatim.
func TestSampleMetricsUsesInjectedProvider(t *testing.T) {
	logger := logrus.New()
	fixed := map[string]interface{}{"cpu_usage": 42.5, "status": "ok"}
	provider := MetricsProvider(func() map[string]interface{} { return fixed })
	handler := NewWebSocketHandler(nil, nil, provider, logger)
	defer handler.Shutdown()

	got := handler.sampleMetrics()
	if got["cpu_usage"] != 42.5 {
		t.Errorf("expected injected cpu_usage 42.5, got %v", got["cpu_usage"])
	}
	if got["status"] != "ok" {
		t.Errorf("expected injected status 'ok', got %v", got["status"])
	}
}

// TestAlertFilters tests the alert filter matching against the {type, data,
// timestamp} AlertMessage shape.
func TestAlertFilters(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	alert := AlertMessage{
		Type: "security_alert",
		Data: map[string]interface{}{
			"severity": "critical",
			"source":   "monitoring",
		},
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

// dialAlerts connects to the alerts endpoint through a passthrough-auth
// router, optionally with a raw query string (e.g. "severity=critical").
func dialAlerts(t *testing.T, serverURL, query string) *websocket.Conn {
	t.Helper()
	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/ws/alerts"
	if query != "" {
		wsURL += "?" + query
	}
	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("failed to dial alerts websocket: %v", err)
	}
	if resp.StatusCode != http.StatusSwitchingProtocols {
		conn.Close()
		t.Fatalf("expected status 101, got %d", resp.StatusCode)
	}
	return conn
}

// TestWebSocketAlertsFanOutMultiClient proves every connected alerts client
// receives a published alert, not just one random winner.
func TestWebSocketAlertsFanOutMultiClient(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, passthroughRoleGuard)
	server := httptest.NewServer(router)
	defer server.Close()

	const n = 4
	conns := make([]*websocket.Conn, n)
	for i := 0; i < n; i++ {
		conns[i] = dialAlerts(t, server.URL, "")
		defer conns[i].Close()
	}
	// Let all clients register before publishing (addAlertClient runs in the
	// read-pump-launching goroutine started right after Upgrade).
	time.Sleep(50 * time.Millisecond)

	handler.PublishAlert(AlertMessage{
		Type:      "security_alert",
		Data:      map[string]interface{}{"severity": "critical", "title": "test alert"},
		Timestamp: time.Now(),
	})

	for i, conn := range conns {
		conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, msg, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("client %d: failed to read alert: %v", i, err)
		}
		var alert AlertMessage
		if err := json.Unmarshal(msg, &alert); err != nil {
			t.Fatalf("client %d: failed to parse alert: %v", i, err)
		}
		if alert.Type != "security_alert" || alert.Data["severity"] != "critical" {
			t.Fatalf("client %d: unexpected alert content: %+v", i, alert)
		}
	}
}

// TestWebSocketAlertsFanOutHonorsSeverityFilter proves per-client severity
// filters are still applied during fan-out.
func TestWebSocketAlertsFanOutHonorsSeverityFilter(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, passthroughRoleGuard)
	server := httptest.NewServer(router)
	defer server.Close()

	criticalConn := dialAlerts(t, server.URL, "severity=critical")
	defer criticalConn.Close()
	infoConn := dialAlerts(t, server.URL, "severity=info")
	defer infoConn.Close()
	time.Sleep(50 * time.Millisecond)

	handler.PublishAlert(AlertMessage{
		Type:      "security_alert",
		Data:      map[string]interface{}{"severity": "critical"},
		Timestamp: time.Now(),
	})

	criticalConn.SetReadDeadline(time.Now().Add(3 * time.Second))
	if _, _, err := criticalConn.ReadMessage(); err != nil {
		t.Fatalf("critical-filtered client should have received the alert: %v", err)
	}

	infoConn.SetReadDeadline(time.Now().Add(300 * time.Millisecond))
	if _, _, err := infoConn.ReadMessage(); err == nil {
		t.Fatal("info-filtered client should not have received a critical alert")
	}
}

// TestWebSocketAlertsSlowClientDisconnectedWithoutBlockingOthers proves
// deliver's drop-and-disconnect path: a stalled client never blocks the fast
// one, and eventually gets disconnected once its bounded send queue fills.
//
// This sandbox's loopback TCP stack does not enforce realistic send/receive
// buffer backpressure (verified empirically: a raw net.Conn with an explicit
// 2KB SO_SNDBUF/SO_RCVBUF never blocks a writer even after 5s of continuous
// writes with no reader draining it), so a real dialed connection can never
// actually stall here. Instead, the "stalled" client is built by hand and
// registered directly: it owns a real server-side *websocket.Conn (so
// Connection.Close() has something legitimate to close and the client-side
// peer observably sees the disconnect) but — unlike every client
// HandleAlertsWebSocket creates — no write pump is ever started for it, so
// nothing drains its send channel; filling it deterministically reproduces
// "client too slow to keep up" without depending on OS buffer timing.
func TestWebSocketAlertsSlowClientDisconnectedWithoutBlockingOthers(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	handler := NewWebSocketHandler(nil, nil, nil, nil, logger)
	defer handler.Shutdown()

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, passthroughRoleGuard)
	server := httptest.NewServer(router)
	defer server.Close()

	fastConn := dialAlerts(t, server.URL, "")
	defer fastConn.Close()
	time.Sleep(20 * time.Millisecond)

	// A tiny standalone upgrade server just to obtain a real server-side
	// *websocket.Conn for the hand-built client's Connection field.
	serverConnCh := make(chan *websocket.Conn, 1)
	rawUpgrader := websocket.Upgrader{}
	rawServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := rawUpgrader.Upgrade(w, r, nil)
		if err == nil {
			serverConnCh <- c
		}
	}))
	defer rawServer.Close()

	stalledClientConn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(rawServer.URL, "http")+"/", nil)
	if err != nil {
		t.Fatalf("dial raw conn for stalled client: %v", err)
	}
	defer stalledClientConn.Close()
	stalledServerConn := <-serverConnCh

	stalledCtx, stalledCancel := context.WithCancel(context.Background())
	stalledClient := &WebSocketClient{
		ID:         "stalled-1",
		Connection: stalledServerConn,
		ClientType: "alerts",
		send:       make(chan []byte, 4), // small cap: fills fast, deterministically
		ctx:        stalledCtx,
		cancel:     stalledCancel,
	}
	stalledClient.setFilters(map[string]interface{}{})

	handler.clientsMutex.Lock()
	handler.alertClients = append(handler.alertClients, stalledClient)
	handler.clientsMutex.Unlock()

	// Fill the stalled client's queue completely; nothing drains it.
	for i := 0; i < cap(stalledClient.send); i++ {
		stalledClient.send <- []byte("x")
	}

	const rounds = 5
	for i := 0; i < rounds; i++ {
		handler.PublishAlert(AlertMessage{
			Type:      "security_alert",
			Data:      map[string]interface{}{"severity": "critical", "seq": i},
			Timestamp: time.Now(),
		})
	}

	// The fast (real) client must receive every alert promptly, unaffected by
	// the stalled client's full queue.
	fastConn.SetReadDeadline(time.Now().Add(3 * time.Second))
	for i := 0; i < rounds; i++ {
		if _, _, err := fastConn.ReadMessage(); err != nil {
			t.Fatalf("fast client failed to read alert %d: %v", i, err)
		}
	}

	// deliver must have disconnected the stalled client the first time it saw
	// the full queue.
	select {
	case <-stalledClient.ctx.Done():
	default:
		t.Fatal("stalled client should have been canceled once its send queue filled")
	}
	stalledClientConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, _, err := stalledClientConn.ReadMessage(); err == nil {
		t.Fatal("stalled client's connection should have been closed by deliver")
	}
}

// TestCheckOrigin table-tests the CSWSH guard.
func TestCheckOrigin(t *testing.T) {
	logger := logrus.New()
	handler := NewWebSocketHandler(nil, nil, AllowedOrigins{"http://good.example", "*wontmatchliteral"}, logger)
	defer handler.Shutdown()

	cases := []struct {
		name   string
		host   string
		origin string
		want   bool
	}{
		{"no origin header allowed (non-browser client)", "api.example", "", true},
		{"same-origin allowed", "api.example", "http://api.example", true},
		{"origin in allow-list allowed", "api.example", "http://good.example", true},
		{"cross-origin not listed denied", "api.example", "http://evil.example", false},
		{"scheme mismatch denied", "api.example", "https://good.example", false},
		{"malformed origin denied", "api.example", "not-a-url", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodGet, "http://"+tc.host+"/ws/alerts", nil)
			r.Host = tc.host
			if tc.origin != "" {
				r.Header.Set("Origin", tc.origin)
			}
			if got := handler.checkOrigin(r); got != tc.want {
				t.Errorf("checkOrigin(host=%s, origin=%s) = %v, want %v", tc.host, tc.origin, got, tc.want)
			}
		})
	}

	// AllowedOrigins containing "*" allows anything.
	wildcard := NewWebSocketHandler(nil, nil, AllowedOrigins{"*"}, logger)
	defer wildcard.Shutdown()
	r := httptest.NewRequest(http.MethodGet, "http://api.example/ws/alerts", nil)
	r.Host = "api.example"
	r.Header.Set("Origin", "http://anything.example")
	if !wildcard.checkOrigin(r) {
		t.Error(`AllowedOrigins containing "*" should allow any origin`)
	}
}

// TestFilterMetrics covers Resolution (a)'s per-client source filtering: a
// matching source narrows the sample, an unmatched one falls back to the
// full sample rather than silently emptying the response, and no sources
// means no filtering.
func TestFilterMetrics(t *testing.T) {
	sample := map[string]interface{}{"timestamp": int64(1), "cpu_usage": 12.5, "memory_usage": 30.0}

	if got := filterMetrics(sample, nil); len(got) != 3 {
		t.Fatalf("no sources filter should return everything, got %+v", got)
	}

	got := filterMetrics(sample, []string{"cpu_usage"})
	if _, ok := got["memory_usage"]; ok {
		t.Fatalf("expected memory_usage filtered out, got %+v", got)
	}
	if got["cpu_usage"] != 12.5 || got["timestamp"] != int64(1) {
		t.Fatalf("expected cpu_usage+timestamp kept, got %+v", got)
	}

	got = filterMetrics(sample, []string{"does-not-exist"})
	if len(got) != len(sample) {
		t.Fatalf("no matching source should fall back to the full sample, got %+v", got)
	}
}
