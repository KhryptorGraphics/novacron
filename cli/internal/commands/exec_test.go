package commands

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

func TestExecSendsCommandOverWebSocket(t *testing.T) {
	withTempHome(t)

	received := make(chan []string, 1)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ws/v1/namespaces/default/vms/vm-1/exec" {
			t.Fatalf("expected exec websocket path, got %s", r.URL.Path)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		var msg struct {
			Command []string `json:"command"`
		}
		if err := conn.ReadJSON(&msg); err != nil {
			t.Fatalf("read exec message: %v", err)
		}
		received <- msg.Command
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	cmd := NewExecCommand()
	cmd.SetArgs([]string{"vm-1", "--", "ls", "-la", "/var/log"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("exec failed: %v", err)
	}

	got := <-received
	wantJSON := `["ls","-la","/var/log"]`
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal received command: %v", err)
	}
	if string(gotJSON) != wantJSON {
		t.Fatalf("expected command %s, got %s", wantJSON, gotJSON)
	}
}

func TestExecRequiresCommand(t *testing.T) {
	cmd := NewExecCommand()
	cmd.SetArgs([]string{"vm-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "requires a VM and command") {
		t.Fatalf("expected command validation error, got %v", err)
	}
}
