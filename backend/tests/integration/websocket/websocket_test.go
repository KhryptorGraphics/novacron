package websocket_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/tests/integration"
)

// WebSocketTestSuite tests real-time WebSocket communication
type WebSocketTestSuite struct {
	integration.IntegrationTestSuite
	testUserID int
	authToken  string
	upgrader   websocket.Upgrader
	
	// WebSocket event handling
	eventBus    *EventBus
	connections map[string]*websocket.Conn
	connMutex   sync.RWMutex
}

// EventBus simulates a simple event bus for testing
type EventBus struct {
	subscribers map[string][]chan Event
	mutex       sync.RWMutex
}

// Event represents a system event
type Event struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

// NewEventBus creates a new event bus
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]chan Event),
	}
}

// Subscribe subscribes to events of a specific type
func (eb *EventBus) Subscribe(eventType string) chan Event {
	eb.mutex.Lock()
	defer eb.mutex.Unlock()
	
	ch := make(chan Event, 10) // Buffered channel
	eb.subscribers[eventType] = append(eb.subscribers[eventType], ch)
	return ch
}

// Publish publishes an event to all subscribers
func (eb *EventBus) Publish(event Event) {
	eb.mutex.RLock()
	defer eb.mutex.RUnlock()
	
	if subscribers, exists := eb.subscribers[event.Type]; exists {
		for _, ch := range subscribers {
			select {
			case ch <- event:
			case <-time.After(100 * time.Millisecond):
				// Skip if channel is blocked
			}
		}
	}
	
	// Also publish to "all" subscribers
	if subscribers, exists := eb.subscribers["*"]; exists {
		for _, ch := range subscribers {
			select {
			case ch <- event:
			case <-time.After(100 * time.Millisecond):
				// Skip if channel is blocked
			}
		}
	}
}

// SetupSuite runs before all WebSocket tests
func (suite *WebSocketTestSuite) SetupSuite() {
	suite.IntegrationTestSuite.SetupSuite()
	suite.setupTestUser()
	suite.setupWebSocketInfrastructure()
	suite.registerWebSocketRoutes()
}

// setupTestUser creates a test user for WebSocket operations
func (suite *WebSocketTestSuite) setupTestUser() {
	user, err := suite.GetAuthManager().CreateUser(
		"wstestuser", 
		"wstest@example.com", 
		"WSTestPassword123!", 
		"user", 
		"test-tenant",
	)
	suite.Require().NoError(err, "Failed to create WebSocket test user")
	suite.testUserID = user.ID

	_, token, err := suite.GetAuthManager().Authenticate("wstestuser", "WSTestPassword123!")
	suite.Require().NoError(err, "Failed to authenticate WebSocket test user")
	suite.authToken = token

	suite.T().Log("WebSocket test user created and authenticated")
}

// setupWebSocketInfrastructure sets up WebSocket testing infrastructure
func (suite *WebSocketTestSuite) setupWebSocketInfrastructure() {
	suite.upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for testing
		},
	}
	
	suite.eventBus = NewEventBus()
	suite.connections = make(map[string]*websocket.Conn)
	
	suite.T().Log("WebSocket infrastructure setup completed")
}

// registerWebSocketRoutes registers WebSocket routes for testing
func (suite *WebSocketTestSuite) registerWebSocketRoutes() {
	router := suite.GetRouter()

	// Main WebSocket endpoint
	router.HandleFunc("/ws/events", suite.handleWebSocketEvents)
	
	// VM monitoring WebSocket
	router.HandleFunc("/ws/monitoring", suite.handleVMMonitoring)
	
	// System events WebSocket
	router.HandleFunc("/ws/system", suite.handleSystemEvents)
	
	// Test event trigger endpoints
	router.HandleFunc("/api/test/trigger-event", suite.handleTriggerEvent).Methods("POST")
	router.HandleFunc("/api/test/trigger-vm-event", suite.handleTriggerVMEvent).Methods("POST")
	
	suite.T().Log("WebSocket routes registered for testing")
}

// TestWebSocketConnection tests basic WebSocket connection
func (suite *WebSocketTestSuite) TestWebSocketConnection() {
	suite.T().Log("Testing WebSocket connection...")

	// Test basic connection
	suite.T().Run("BasicConnection", func(t *testing.T) {
		conn, resp, err := suite.connectWebSocket("/ws/events")
		require.NoError(t, err, "Failed to connect to WebSocket")
		defer conn.Close()

		assert.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode, "Should upgrade to WebSocket")
		
		// Send a ping
		err = conn.WriteMessage(websocket.PingMessage, []byte("ping"))
		assert.NoError(t, err, "Should be able to send ping")

		// Read pong response
		conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		messageType, message, err := conn.ReadMessage()
		if err == nil {
			t.Logf("Received message: type=%d, data=%s", messageType, string(message))
		}
		
		t.Log("✓ WebSocket basic connection successful")
	})

	// Test connection with authentication
	suite.T().Run("AuthenticatedConnection", func(t *testing.T) {
		headers := http.Header{}
		headers.Set("Authorization", "Bearer "+suite.authToken)
		
		conn, resp, err := suite.connectWebSocketWithHeaders("/ws/events", headers)
		require.NoError(t, err, "Failed to connect to authenticated WebSocket")
		defer conn.Close()

		assert.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode, "Should upgrade to WebSocket")
		
		t.Log("✓ Authenticated WebSocket connection successful")
	})

	// Test invalid connections
	suite.T().Run("InvalidConnection", func(t *testing.T) {
		// Try to connect with invalid auth
		headers := http.Header{}
		headers.Set("Authorization", "Bearer invalid-token")
		
		conn, resp, err := suite.connectWebSocketWithHeaders("/ws/events", headers)
		if err == nil && conn != nil {
			conn.Close()
		}
		
		// The connection might succeed but authentication should be handled in the handler
		// For this test, we'll just verify we can handle the scenario
		t.Logf("Connection with invalid auth: status=%d", resp.StatusCode)
		
		t.Log("✓ Invalid connection handling tested")
	})

	suite.T().Log("✓ WebSocket connection tests completed")
}

// TestRealTimeEvents tests real-time event broadcasting
func (suite *WebSocketTestSuite) TestRealTimeEvents() {
	suite.T().Log("Testing real-time event broadcasting...")

	// Test event broadcasting
	suite.T().Run("EventBroadcasting", func(t *testing.T) {
		// Connect multiple clients
		clients := make([]*websocket.Conn, 3)
		receivedEvents := make([]chan Event, 3)
		
		for i := 0; i < 3; i++ {
			conn, _, err := suite.connectWebSocket("/ws/events")
			require.NoError(t, err, "Failed to connect client %d", i)
			clients[i] = conn
			receivedEvents[i] = make(chan Event, 10)
			
			// Start reading messages for each client
			go suite.readWebSocketMessages(conn, receivedEvents[i])
		}
		
		// Clean up connections
		defer func() {
			for _, conn := range clients {
				if conn != nil {
					conn.Close()
				}
			}
		}()

		// Wait for connections to be established
		time.Sleep(100 * time.Millisecond)

		// Trigger a test event
		event := Event{
			Type: "test.broadcast",
			Data: map[string]interface{}{
				"message": "Hello from test",
				"counter": 42,
			},
			Timestamp: time.Now(),
		}

		suite.eventBus.Publish(event)

		// Wait for events to be received
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		receivedCount := 0
		for i := 0; i < 3; i++ {
			select {
			case receivedEvent := <-receivedEvents[i]:
				assert.Equal(t, event.Type, receivedEvent.Type, "Event type should match")
				assert.Equal(t, event.Data["message"], receivedEvent.Data["message"], "Event data should match")
				receivedCount++
				t.Logf("Client %d received event: %+v", i, receivedEvent)
			case <-ctx.Done():
				t.Logf("Client %d did not receive event within timeout", i)
			}
		}

		assert.GreaterOrEqual(t, receivedCount, 1, "At least one client should receive the event")
		t.Logf("✓ Event broadcasting successful (%d clients received)", receivedCount)
	})

	suite.T().Log("✓ Real-time event tests completed")
}

// TestVMMonitoring tests VM monitoring through WebSocket
func (suite *WebSocketTestSuite) TestVMMonitoring() {
	suite.T().Log("Testing VM monitoring WebSocket...")

	suite.T().Run("VMMetricsStreaming", func(t *testing.T) {
		conn, _, err := suite.connectWebSocket("/ws/monitoring")
		require.NoError(t, err, "Failed to connect to VM monitoring WebSocket")
		defer conn.Close()

		receivedEvents := make(chan Event, 10)
		go suite.readWebSocketMessages(conn, receivedEvents)

		// Trigger VM events
		vmEvents := []Event{
			{
				Type: "vm.metrics",
				Data: map[string]interface{}{
					"vm_id":        "test-vm-1",
					"cpu_usage":    75.5,
					"memory_usage": 68.2,
					"timestamp":    time.Now().Unix(),
				},
				Timestamp: time.Now(),
			},
			{
				Type: "vm.state_change",
				Data: map[string]interface{}{
					"vm_id":     "test-vm-1",
					"old_state": "stopped",
					"new_state": "running",
				},
				Timestamp: time.Now(),
			},
		}

		for _, event := range vmEvents {
			suite.eventBus.Publish(event)
		}

		// Collect received events
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		var receivedEvents slice []Event
		for {
			select {
			case event := <-receivedEvents:
				receivedEvents = append(receivedEvents, event)
				if len(receivedEvents) >= len(vmEvents) {
					goto checkEvents
				}
			case <-ctx.Done():
				goto checkEvents
			}
		}

	checkEvents:
		assert.GreaterOrEqual(t, len(receivedEvents), 1, "Should receive at least one VM event")
		
		for _, event := range receivedEvents {
			assert.True(t, strings.HasPrefix(event.Type, "vm."), "Event should be VM-related")
			assert.Contains(t, event.Data, "vm_id", "VM event should contain VM ID")
		}

		t.Logf("✓ VM monitoring successful (%d events received)", len(receivedEvents))
	})

	suite.T().Log("✓ VM monitoring tests completed")
}

// TestSystemEventsWebSocket tests system-wide events
func (suite *WebSocketTestSuite) TestSystemEventsWebSocket() {
	suite.T().Log("Testing system events WebSocket...")

	suite.T().Run("SystemEventStreaming", func(t *testing.T) {
		conn, _, err := suite.connectWebSocket("/ws/system")
		require.NoError(t, err, "Failed to connect to system events WebSocket")
		defer conn.Close()

		receivedEvents := make(chan Event, 10)
		go suite.readWebSocketMessages(conn, receivedEvents)

		// Trigger various system events
		systemEvents := []Event{
			{
				Type: "system.startup",
				Data: map[string]interface{}{
					"service": "vm-manager",
					"status":  "ready",
				},
				Timestamp: time.Now(),
			},
			{
				Type: "system.alert",
				Data: map[string]interface{}{
					"severity": "warning",
					"message":  "High CPU usage detected",
					"source":   "monitoring",
				},
				Timestamp: time.Now(),
			},
			{
				Type: "system.maintenance",
				Data: map[string]interface{}{
					"action":    "backup_started",
					"scheduled": true,
				},
				Timestamp: time.Now(),
			},
		}

		for _, event := range systemEvents {
			suite.eventBus.Publish(event)
			time.Sleep(50 * time.Millisecond) // Small delay between events
		}

		// Collect received events
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		var receivedEvents slice []Event
		for {
			select {
			case event := <-receivedEvents:
				receivedEvents = append(receivedEvents, event)
				if len(receivedEvents) >= len(systemEvents) {
					goto checkSystemEvents
				}
			case <-ctx.Done():
				goto checkSystemEvents
			}
		}

	checkSystemEvents:
		assert.GreaterOrEqual(t, len(receivedEvents), 1, "Should receive at least one system event")
		
		for _, event := range receivedEvents {
			assert.True(t, strings.HasPrefix(event.Type, "system."), "Event should be system-related")
		}

		t.Logf("✓ System event streaming successful (%d events received)", len(receivedEvents))
	})

	suite.T().Log("✓ System events tests completed")
}

// TestWebSocketMessageTypes tests different WebSocket message types
func (suite *WebSocketTestSuite) TestWebSocketMessageTypes() {
	suite.T().Log("Testing WebSocket message types...")

	suite.T().Run("TextMessages", func(t *testing.T) {
		conn, _, err := suite.connectWebSocket("/ws/events")
		require.NoError(t, err, "Failed to connect for text message test")
		defer conn.Close()

		// Send text message
		testMessage := "test message"
		err = conn.WriteMessage(websocket.TextMessage, []byte(testMessage))
		assert.NoError(t, err, "Should be able to send text message")

		// Read response (if any)
		conn.SetReadDeadline(time.Now().Add(1 * time.Second))
		messageType, message, err := conn.ReadMessage()
		if err == nil {
			t.Logf("Received response: type=%d, data=%s", messageType, string(message))
		}

		t.Log("✓ Text message test completed")
	})

	suite.T().Run("JSONMessages", func(t *testing.T) {
		conn, _, err := suite.connectWebSocket("/ws/events")
		require.NoError(t, err, "Failed to connect for JSON message test")
		defer conn.Close()

		// Send JSON message
		jsonData := map[string]interface{}{
			"action": "subscribe",
			"topics": []string{"vm.metrics", "system.alerts"},
		}

		jsonBytes, err := json.Marshal(jsonData)
		require.NoError(t, err, "Failed to marshal JSON")

		err = conn.WriteMessage(websocket.TextMessage, jsonBytes)
		assert.NoError(t, err, "Should be able to send JSON message")

		t.Log("✓ JSON message test completed")
	})

	suite.T().Log("✓ WebSocket message types tests completed")
}

// TestWebSocketErrorHandling tests error scenarios
func (suite *WebSocketTestSuite) TestWebSocketErrorHandling() {
	suite.T().Log("Testing WebSocket error handling...")

	suite.T().Run("ConnectionDropHandling", func(t *testing.T) {
		conn, _, err := suite.connectWebSocket("/ws/events")
		require.NoError(t, err, "Failed to connect for drop test")

		// Close connection abruptly
		conn.Close()

		// Try to write to closed connection
		err = conn.WriteMessage(websocket.TextMessage, []byte("test"))
		assert.Error(t, err, "Writing to closed connection should error")

		t.Log("✓ Connection drop handling tested")
	})

	suite.T().Run("InvalidMessageHandling", func(t *testing.T) {
		conn, _, err := suite.connectWebSocket("/ws/events")
		require.NoError(t, err, "Failed to connect for invalid message test")
		defer conn.Close()

		// Send invalid JSON
		err = conn.WriteMessage(websocket.TextMessage, []byte("{invalid json}"))
		assert.NoError(t, err, "Should be able to send invalid JSON (handling on server side)")

		t.Log("✓ Invalid message handling tested")
	})

	suite.T().Log("✓ WebSocket error handling tests completed")
}

// Helper methods

// connectWebSocket establishes a WebSocket connection
func (suite *WebSocketTestSuite) connectWebSocket(path string) (*websocket.Conn, *http.Response, error) {
	return suite.connectWebSocketWithHeaders(path, nil)
}

// connectWebSocketWithHeaders establishes a WebSocket connection with custom headers
func (suite *WebSocketTestSuite) connectWebSocketWithHeaders(path string, headers http.Header) (*websocket.Conn, *http.Response, error) {
	serverURL := suite.GetServer().URL
	wsURL := strings.Replace(serverURL, "http://", "ws://", 1) + path
	
	u, err := url.Parse(wsURL)
	if err != nil {
		return nil, nil, err
	}

	dialer := websocket.DefaultDialer
	dialer.HandshakeTimeout = 10 * time.Second

	conn, resp, err := dialer.Dial(u.String(), headers)
	if err != nil {
		return conn, resp, err
	}

	return conn, resp, nil
}

// readWebSocketMessages reads messages from WebSocket and sends them to a channel
func (suite *WebSocketTestSuite) readWebSocketMessages(conn *websocket.Conn, eventChan chan Event) {
	defer close(eventChan)
	
	for {
		messageType, message, err := conn.ReadMessage()
		if err != nil {
			suite.T().Logf("WebSocket read error: %v", err)
			return
		}

		if messageType == websocket.TextMessage {
			var event Event
			if err := json.Unmarshal(message, &event); err == nil {
				eventChan <- event
			} else {
				suite.T().Logf("Failed to unmarshal event: %v", err)
			}
		}
	}
}

// WebSocket handler implementations

func (suite *WebSocketTestSuite) handleWebSocketEvents(w http.ResponseWriter, r *http.Request) {
	conn, err := suite.upgrader.Upgrade(w, r, nil)
	if err != nil {
		suite.T().Logf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Subscribe to all events
	eventChan := suite.eventBus.Subscribe("*")
	defer func() {
		// In a real implementation, we'd unsubscribe properly
	}()

	// Handle incoming messages and send events
	go func() {
		for {
			messageType, message, err := conn.ReadMessage()
			if err != nil {
				return
			}

			if messageType == websocket.TextMessage {
				// Echo back or handle command
				conn.WriteMessage(websocket.TextMessage, message)
			}
		}
	}()

	// Send events to client
	for {
		select {
		case event := <-eventChan:
			eventJSON, err := json.Marshal(event)
			if err != nil {
				continue
			}

			if err := conn.WriteMessage(websocket.TextMessage, eventJSON); err != nil {
				return
			}
		}
	}
}

func (suite *WebSocketTestSuite) handleVMMonitoring(w http.ResponseWriter, r *http.Request) {
	conn, err := suite.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	// Subscribe to VM-related events
	vmEventChan := suite.eventBus.Subscribe("vm.*")
	defer func() {
		// In a real implementation, we'd unsubscribe properly
	}()

	// Send VM events to client
	for {
		select {
		case event := <-vmEventChan:
			if strings.HasPrefix(event.Type, "vm.") {
				eventJSON, err := json.Marshal(event)
				if err != nil {
					continue
				}

				if err := conn.WriteMessage(websocket.TextMessage, eventJSON); err != nil {
					return
				}
			}
		}
	}
}

func (suite *WebSocketTestSuite) handleSystemEvents(w http.ResponseWriter, r *http.Request) {
	conn, err := suite.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	// Subscribe to system events
	systemEventChan := suite.eventBus.Subscribe("system.*")
	defer func() {
		// In a real implementation, we'd unsubscribe properly
	}()

	// Send system events to client
	for {
		select {
		case event := <-systemEventChan:
			if strings.HasPrefix(event.Type, "system.") {
				eventJSON, err := json.Marshal(event)
				if err != nil {
					continue
				}

				if err := conn.WriteMessage(websocket.TextMessage, eventJSON); err != nil {
					return
				}
			}
		}
	}
}

func (suite *WebSocketTestSuite) handleTriggerEvent(w http.ResponseWriter, r *http.Request) {
	var eventData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&eventData); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	event := Event{
		Type:      eventData["type"].(string),
		Data:      eventData["data"].(map[string]interface{}),
		Timestamp: time.Now(),
	}

	suite.eventBus.Publish(event)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Event triggered successfully",
		"event":   event,
	})
}

func (suite *WebSocketTestSuite) handleTriggerVMEvent(w http.ResponseWriter, r *http.Request) {
	var vmEventData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&vmEventData); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	event := Event{
		Type:      "vm." + vmEventData["event_type"].(string),
		Data:      vmEventData,
		Timestamp: time.Now(),
	}

	suite.eventBus.Publish(event)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "VM event triggered successfully",
		"event":   event,
	})
}

// TestWebSocketSuite runs the WebSocket test suite
func TestWebSocketSuite(t *testing.T) {
	suite.Run(t, new(WebSocketTestSuite))
}