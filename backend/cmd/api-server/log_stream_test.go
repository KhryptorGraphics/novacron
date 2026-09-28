package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"

	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// TestCanonicalLogStreamDeliversGlobalLoggerEntries: a line logged through
// the api-server's global logger reaches an /api/ws/logs client via the sink
// initializeCanonicalServices installs, and LOG_STREAM_LEVEL filters: an info
// line (emitted by the info-level logger) is not streamed at stream level
// warn, the warn line after it is.
func TestCanonicalLogStreamDeliversGlobalLoggerEntries(t *testing.T) {
	cfg := &config.Config{}
	cfg.VM.StoragePath = t.TempDir()
	cfg.Logging.StreamLevel = "warn"

	// Unconnected handle: service construction never touches the DB.
	db, err := sql.Open("postgres", "postgres://127.0.0.1:1/none?sslmode=disable")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	services, err := initializeCanonicalServices(cfg, db, authManager)
	if err != nil {
		t.Fatalf("initializeCanonicalServices: %v", err)
	}
	defer services.shutdown()

	router := mux.NewRouter()
	services.websocketHandler.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return wsAuthMiddleware(authManager, nil)(requireRoleHandler(required, next))
	})
	server := httptest.NewServer(router)
	defer server.Close()

	token := strings.TrimPrefix(signedBearerToken(t, authManager, "7", "default", "admin"), "Bearer ")
	d := websocket.Dialer{Subprotocols: []string{"bearer", token}}
	conn, resp, err := d.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/api/ws/logs", nil)
	if err != nil {
		t.Fatalf("dial /api/ws/logs: %v (resp=%v)", err, resp)
	}
	defer conn.Close()
	time.Sleep(50 * time.Millisecond) // let the client register

	logger.Info("log-stream-test: below stream level")
	logger.Warn("log-stream-test: peer unreachable", "vm", "vm-7", "node", "node-2")

	// Other code may log through the global logger meanwhile; skip foreign
	// lines, but the filtered info line must never arrive.
	for {
		conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, raw, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("read log message: %v", err)
		}
		var msg websocketapi.LogMessage
		if err := json.Unmarshal(raw, &msg); err != nil {
			t.Fatalf("parse log message %s: %v", raw, err)
		}
		switch msg.Message {
		case "log-stream-test: below stream level":
			t.Fatalf("info line streamed despite LOG_STREAM_LEVEL=warn: %+v", msg)
		case "log-stream-test: peer unreachable":
			if msg.Level != "warn" || msg.Source != "system" || msg.Component != "api-server" {
				t.Fatalf("level/source/component = %q/%q/%q, want warn/system/api-server", msg.Level, msg.Source, msg.Component)
			}
			if msg.VMID != "vm-7" || msg.Labels["node"] != "node-2" {
				t.Fatalf("vm_id/labels = %q/%v, want vm-7 and node=node-2", msg.VMID, msg.Labels)
			}
			return
		}
	}
}
