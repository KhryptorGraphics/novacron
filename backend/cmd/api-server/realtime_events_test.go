package main

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

	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/healing"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func dialWSAlerts(t *testing.T, serverURL string, protocols []string) (*websocket.Conn, *http.Response, error) {
	t.Helper()
	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/api/ws/alerts"
	d := websocket.Dialer{Subprotocols: protocols}
	return d.Dial(wsURL, nil)
}

// TestRealtimeEventBridgeDeliversHealthDegradedToTwoAlertClients is the
// acceptance-criterion smoke test: a healing.EventTypeHealthDegraded event
// published on a real InProcessEventBus reaches two independently-connected
// /api/ws/alerts clients authenticated ONLY via Sec-WebSocket-Protocol
// (no Authorization header), proving the browser WS auth adapter + fan-out
// work end to end.
func TestRealtimeEventBridgeDeliversHealthDegradedToTwoAlertClients(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	ws := websocketapi.NewWebSocketHandler(nil, nil, logger)
	defer ws.Shutdown()
	bus := events.NewInProcessEventBus(logger)
	bridge := newRealtimeEventBridge(ws, logger)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bridge.subscribe(ctx, bus)

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	ws.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return wsAuthMiddleware(authManager, nil)(requireRoleHandler(required, next))
	})
	server := httptest.NewServer(router)
	defer server.Close()

	token := strings.TrimPrefix(signedBearerToken(t, authManager, "7", "default", "admin"), "Bearer ")

	conn1, resp1, err := dialWSAlerts(t, server.URL, []string{"bearer", token})
	if err != nil {
		t.Fatalf("client 1 dial: %v (resp=%v)", err, resp1)
	}
	defer conn1.Close()
	if got := resp1.Header.Get("Sec-WebSocket-Protocol"); got != "bearer" {
		t.Fatalf("expected upgrader to echo the bearer subprotocol, got %q", got)
	}

	conn2, resp2, err := dialWSAlerts(t, server.URL, []string{"bearer", token})
	if err != nil {
		t.Fatalf("client 2 dial: %v (resp=%v)", err, resp2)
	}
	defer conn2.Close()

	time.Sleep(50 * time.Millisecond) // let both clients register

	if err := bus.Publish(ctx, &events.OrchestrationEvent{
		Type:      events.EventType(healing.EventTypeHealthDegraded),
		Target:    "vm-1",
		Timestamp: time.Now(),
		Data:      map[string]interface{}{},
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}

	for i, conn := range []*websocket.Conn{conn1, conn2} {
		conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, msg, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("client %d failed to read alert: %v", i, err)
		}
		var alert websocketapi.AlertMessage
		if err := json.Unmarshal(msg, &alert); err != nil {
			t.Fatalf("client %d failed to parse alert: %v", i, err)
		}
		if alert.Type != "security_alert" {
			t.Fatalf("client %d: expected security_alert, got %+v", i, alert)
		}
	}
}

// TestWSAuthMiddlewareRejectsMissingInvalidPendingTwoFA proves the
// integration wiring (not authenticateToken's internal rules, covered by
// auth_session_test.go): a WS upgrade with no bearer subprotocol, a garbage
// token, or a pending_2fa token is rejected before the upgrade completes.
func TestWSAuthMiddlewareRejectsMissingInvalidPendingTwoFA(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	ws := websocketapi.NewWebSocketHandler(nil, nil, logrus.New())
	defer ws.Shutdown()
	ws.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return wsAuthMiddleware(authManager, nil)(requireRoleHandler(required, next))
	})
	server := httptest.NewServer(router)
	defer server.Close()

	pending, err := issuePending2FAToken(authManager.GetJWTSecret(), &auth.User{
		ID: "9", Email: "user9@example.com", Username: "user9", RoleIDs: []string{"admin"}, TenantID: "default",
	})
	if err != nil {
		t.Fatalf("issue pending 2FA token: %v", err)
	}

	cases := []struct {
		name      string
		protocols []string
	}{
		{"missing token", nil},
		{"invalid token", []string{"bearer", "not-a-real-jwt"}},
		{"pending_2fa token", []string{"bearer", pending}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conn, resp, err := dialWSAlerts(t, server.URL, tc.protocols)
			if err == nil {
				conn.Close()
				t.Fatal("expected the dial to fail")
			}
			if resp == nil || resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("expected 401, got resp=%v err=%v", resp, err)
			}
		})
	}
}

// TestRealtimeVMListenerPublishesVMStatusAndErrorAlert proves a VMEventError
// produces BOTH a vm_status message and a security_alert on the alerts
// channel.
func TestRealtimeVMListenerPublishesVMStatusAndErrorAlert(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	ws := websocketapi.NewWebSocketHandler(nil, nil, logger)
	defer ws.Shutdown()

	router := mux.NewRouter()
	router.HandleFunc("/ws/alerts", ws.HandleAlertsWebSocket).Methods("GET")
	server := httptest.NewServer(router)
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws/alerts"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	time.Sleep(20 * time.Millisecond)

	listener := newRealtimeVMListener(ws)
	vm, err := core_vm.NewVM(core_vm.VMConfig{ID: "vm-err", Name: "vm-err", Type: core_vm.VMTypeProcess, Command: "/bin/true"})
	if err != nil {
		t.Fatalf("NewVM: %v", err)
	}
	vm.SetState(core_vm.StateFailed)
	listener.OnVMEvent(core_vm.VMEvent{Type: core_vm.VMEventError, VM: vm, Message: "process died"})

	sawVMStatus, sawAlert := false, false
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	for i := 0; i < 2; i++ {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
		var alert websocketapi.AlertMessage
		if err := json.Unmarshal(msg, &alert); err != nil {
			t.Fatalf("parse %d: %v", i, err)
		}
		switch alert.Type {
		case "vm_status":
			sawVMStatus = true
			if alert.Data["id"] != "vm-err" {
				t.Fatalf("vm_status: unexpected id %v", alert.Data["id"])
			}
		case "security_alert":
			sawAlert = true
		}
	}
	if !sawVMStatus || !sawAlert {
		t.Fatalf("expected both vm_status and security_alert, got vm_status=%v security_alert=%v", sawVMStatus, sawAlert)
	}
}

func TestAlertStoreListNewestFirstAndBounded(t *testing.T) {
	s := newAlertStore(3)
	for i := 0; i < 5; i++ {
		s.add(storedAlert{Type: "t", Data: map[string]interface{}{"i": i}, Timestamp: time.Now()})
	}
	list := s.list()
	if len(list) != 3 {
		t.Fatalf("expected bounded length 3, got %d", len(list))
	}
	// Newest-first: last added (i=4) first, oldest kept (i=2) last.
	if list[0].Data["i"] != 4 || list[2].Data["i"] != 2 {
		t.Fatalf("expected newest-first order [4,3,2], got %+v", list)
	}
}
