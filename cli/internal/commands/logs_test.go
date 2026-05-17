package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
)

func TestLogsCommandStreamsWebSocketLogs(t *testing.T) {
	withTempHome(t)

	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ws/logs/system" {
			t.Fatalf("expected system logs path, got %s", r.URL.Path)
		}
		if r.URL.Query().Get("level") != "error" || r.URL.Query().Get("components") != "scheduler" {
			t.Fatalf("unexpected query: %s", r.URL.RawQuery)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		_ = conn.WriteJSON(logStreamMessage{
			Type:      "log",
			Source:    "system",
			Level:     "error",
			Message:   "scheduler failed placement",
			Timestamp: time.Date(2026, 5, 17, 0, 30, 0, 0, time.UTC),
			Component: "scheduler",
		})
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewLogsCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"system", "--count", "1", "--level", "error", "--components", "scheduler"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("logs command failed: %v", err)
	}

	for _, expected := range []string{"TIME", "SOURCE", "LEVEL", "COMPONENT", "MESSAGE", "system", "error", "scheduler failed placement"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected logs output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestLogsCommandAppliesStoredTokenToWebSocket(t *testing.T) {
	withTempHome(t)

	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer stored-token" {
			t.Fatalf("expected stored bearer token, got %q", got)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		_ = conn.WriteJSON(logStreamMessage{
			Source:    "audit",
			Level:     "info",
			Message:   "token checked",
			Timestamp: time.Date(2026, 5, 17, 0, 31, 0, 0, time.UTC),
		})
	}))
	defer server.Close()

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{
		Name:      "prod",
		Server:    server.URL,
		Namespace: "default",
		AuthType:  "token",
	}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}

	store, err := auth.NewTokenStore()
	if err != nil {
		t.Fatalf("new token store: %v", err)
	}
	if err := store.Save("prod", &auth.TokenAuth{
		Token:        "stored-token",
		RefreshToken: "stored-refresh",
		ExpiresAt:    time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatalf("save token: %v", err)
	}

	var output bytes.Buffer
	cmd := NewLogsCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"audit", "--count", "1"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("logs command failed: %v", err)
	}
	if !strings.Contains(output.String(), "token checked") {
		t.Fatalf("expected streamed log output, got:\n%s", output.String())
	}
}

func TestLogsCommandRejectsBadCount(t *testing.T) {
	cmd := NewLogsCommand()
	cmd.SetArgs([]string{"--count", "0"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "count must be greater than zero") {
		t.Fatalf("expected count validation error, got %v", err)
	}
}

func TestLogStreamMessageJSONShape(t *testing.T) {
	data := []byte(`{"source":"vm","level":"warn","message":"cpu high","timestamp":"2026-05-17T00:32:00Z","component":"agent","vm_id":"vm-1"}`)
	var msg logStreamMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("decode log stream message: %v", err)
	}
	if msg.Source != "vm" || msg.VMID != "vm-1" || msg.Message != "cpu high" {
		t.Fatalf("unexpected message: %#v", msg)
	}
}
