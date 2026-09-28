package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"

	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/autoscaling"
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

// pollUntil fails the test unless cond turns true within 3s.
func pollUntil(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal(msg)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func hasRecentAlert(title string) bool {
	for _, a := range recentAlerts.list() {
		if a.Data["title"] == title {
			return true
		}
	}
	return false
}

// TestHeartbeatPeerFailureReachesEngineAndAlerts runs the node-failure path
// api-server wires: peerLiveness publishes node.failure on the shared bus
// only after peerFailureThreshold consecutive failed heartbeat probes, and
// only once; the orchestration engine marks the peer unhealthy and the
// realtime bridge raises an alert. The next good probe publishes
// node.recovered and the engine marks the peer healthy again.
func TestHeartbeatPeerFailureReachesEngineAndAlerts(t *testing.T) {
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "")
	t.Setenv("NOVACRON_PROBE_BYTES", "0")
	const peerID = "peer-hb-fail"

	var up atomic.Bool
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !up.Load() || r.URL.Path != "/internal/cluster/capacity" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_ = json.NewEncoder(w).Encode(NodeCapacity{NodeID: peerID, Reachable: true})
	}))
	defer peer.Close()

	vmManager := newStubVMManager(t)
	defer vmManager.Stop()
	vmManager.RegisterMigrationPeer(peerID, joinTestRequestAddr(t, peer.URL))

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bus := events.NewInProcessEventBus(logger)
	engine := orchestration.NewDefaultOrchestrationEngine(logger)
	engine.SetEventBus(bus)
	if err := engine.Start(ctx); err != nil {
		t.Fatalf("engine start: %v", err)
	}
	defer engine.Stop(context.Background())
	ws := websocketapi.NewWebSocketHandler(nil, nil, logger)
	defer ws.Shutdown()
	newRealtimeEventBridge(ws, logger).subscribe(ctx, bus)

	// One subscription sees node lifecycle events and a marker in publish
	// order, so "nothing published by this beat" is checked, not slept on.
	const marker = events.EventType("test.marker")
	seen := make(chan events.EventType, 16)
	if _, err := bus.Subscribe(ctx, []events.EventType{events.EventTypeNodeFailure, events.EventTypeNodeRecovered, marker},
		events.NewEventHandlerFunc("recorder", "recorder", func(_ context.Context, e *events.OrchestrationEvent) error {
			seen <- e.Type
			return nil
		})); err != nil {
		t.Fatalf("subscribe: %v", err)
	}
	liveness := newPeerLiveness(bus)
	// beat runs one heartbeat and returns what it published, in order.
	beat := func() []events.EventType {
		beatOnce(ctx, db, vmManager, liveness)
		if err := bus.Publish(ctx, &events.OrchestrationEvent{Type: marker}); err != nil {
			t.Fatalf("publish marker: %v", err)
		}
		var out []events.EventType
		for {
			select {
			case typ := <-seen:
				if typ == marker {
					return out
				}
				out = append(out, typ)
			case <-time.After(3 * time.Second):
				t.Fatal("marker never delivered")
			}
		}
	}

	for i := 1; i < peerFailureThreshold; i++ {
		if got := beat(); len(got) != 0 {
			t.Fatalf("failed probe %d of %d published %v", i, peerFailureThreshold, got)
		}
	}
	if got := beat(); len(got) != 1 || got[0] != events.EventTypeNodeFailure {
		t.Fatalf("probe %d published %v, want [node.failure]", peerFailureThreshold, got)
	}
	if got := beat(); len(got) != 0 {
		t.Fatalf("a still-failed peer published %v again", got)
	}
	pollUntil(t, func() bool {
		st, ok := engine.GetNodeStatuses()[peerID]
		return ok && !st.Healthy
	}, "engine never marked the failed peer unhealthy")
	pollUntil(t, func() bool { return hasRecentAlert("Node unreachable: " + peerID) }, "no node-unreachable alert")

	up.Store(true)
	mock.ExpectExec("INSERT INTO cluster_peers").WillReturnResult(sqlmock.NewResult(0, 1))
	if got := beat(); len(got) != 1 || got[0] != events.EventTypeNodeRecovered {
		t.Fatalf("recovered probe published %v, want [node.recovered]", got)
	}
	pollUntil(t, func() bool { return engine.GetNodeStatuses()[peerID].Healthy }, "engine never marked the recovered peer healthy")
	pollUntil(t, func() bool { return hasRecentAlert("Node reachable: " + peerID) }, "no node-reachable alert")
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("recovered peer's heartbeat was not persisted: %v", err)
	}
}

// TestRealtimeBridgeAlertsOnTriggeredScaling: a real scale-up decision from
// the autoscaler reaches the alerts store through scaling.triggered.
func TestRealtimeBridgeAlertsOnTriggeredScaling(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bus := events.NewInProcessEventBus(logger)
	ws := websocketapi.NewWebSocketHandler(nil, nil, logger)
	defer ws.Shutdown()
	newRealtimeEventBridge(ws, logger).subscribe(ctx, bus)

	scaler := autoscaling.NewDefaultAutoScaler(logger, bus)
	if err := scaler.SetMetricsSource(func() (*autoscaling.MetricsData, error) {
		return &autoscaling.MetricsData{Timestamp: time.Now(), TargetID: "host", TargetType: "node", CPUUsage: 0.95, MemoryUsage: 0.5, ActiveVMs: 2}, nil
	}); err != nil {
		t.Fatalf("SetMetricsSource: %v", err)
	}
	if err := scaler.AddTarget(&autoscaling.AutoScalerTarget{ID: "scale-alert-target", Type: "vm", Enabled: true}); err != nil {
		t.Fatalf("AddTarget: %v", err)
	}
	decision, err := scaler.GetScalingDecision("scale-alert-target")
	if err != nil || decision.Action != autoscaling.ScalingActionScaleUp {
		t.Fatalf("decision = %+v, %v; want scale_up", decision, err)
	}
	want := "Scaling scale_up: scale-alert-target -> " + strconv.Itoa(decision.TargetScale)
	pollUntil(t, func() bool { return hasRecentAlert(want) }, "no alert "+want)
}
