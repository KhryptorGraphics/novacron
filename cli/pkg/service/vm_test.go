package service

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/novacron/cli/pkg/api"
)

func TestVMServiceExecStreamsOutputAndExitStatus(t *testing.T) {
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

		var command struct {
			Command []string `json:"command"`
		}
		if err := conn.ReadJSON(&command); err != nil {
			t.Fatalf("read command: %v", err)
		}
		received <- command.Command

		for _, msg := range []map[string]interface{}{
			{"stream": "stdout", "data": "hello\n"},
			{"stream": "stderr", "data": "warn\n"},
			{"exitCode": 0},
		} {
			if err := conn.WriteJSON(msg); err != nil {
				t.Fatalf("write exec frame: %v", err)
			}
		}
	}))
	defer server.Close()

	client, err := api.NewClient(server.URL)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	var stdout, stderr bytes.Buffer

	err = NewVMService(client).Exec(context.Background(), "default", "vm-1", []string{"echo", "hello"}, nil, &stdout, &stderr)
	if err != nil {
		t.Fatalf("exec failed: %v", err)
	}

	gotCommand, err := json.Marshal(<-received)
	if err != nil {
		t.Fatalf("marshal received command: %v", err)
	}
	if string(gotCommand) != `["echo","hello"]` {
		t.Fatalf("unexpected command: %s", gotCommand)
	}
	if stdout.String() != "hello\n" {
		t.Fatalf("expected stdout stream, got %q", stdout.String())
	}
	if stderr.String() != "warn\n" {
		t.Fatalf("expected stderr stream, got %q", stderr.String())
	}
}

func TestVMServiceExecReturnsNonZeroExitStatus(t *testing.T) {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		var command map[string]interface{}
		if err := conn.ReadJSON(&command); err != nil {
			t.Fatalf("read command: %v", err)
		}
		if err := conn.WriteJSON(map[string]interface{}{"exit_code": 7}); err != nil {
			t.Fatalf("write exit frame: %v", err)
		}
	}))
	defer server.Close()

	client, err := api.NewClient(server.URL)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	err = NewVMService(client).Exec(context.Background(), "default", "vm-1", []string{"false"}, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "exit status 7") {
		t.Fatalf("expected non-zero exit error, got %v", err)
	}
}
