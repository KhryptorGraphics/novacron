package commands

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

func TestVMConsoleStreamsInputAndOutput(t *testing.T) {
	withTempHome(t)

	received := make(chan string, 1)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ws/v1/namespaces/default/vms/vm-1/console" {
			t.Fatalf("expected console websocket path, got %s", r.URL.Path)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		if err := conn.WriteJSON(map[string]interface{}{
			"type":       "output",
			"vm_id":      "vm-1",
			"session_id": "session-1",
			"data":       "login: ",
		}); err != nil {
			t.Fatalf("write console output: %v", err)
		}

		_, input, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("read console input: %v", err)
		}
		received <- string(input)

		if err := conn.WriteJSON(map[string]interface{}{
			"type": "output",
			"data": "accepted\n",
		}); err != nil {
			t.Fatalf("write accepted output: %v", err)
		}
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := newVMConsoleCommand()
	cmd.SetArgs([]string{"vm-1"})
	cmd.SetIn(strings.NewReader("root\n"))
	cmd.SetOut(&output)
	cmd.SetErr(&bytes.Buffer{})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("console failed: %v", err)
	}
	if got := <-received; got != "root\n" {
		t.Fatalf("expected console input %q, got %q", "root\n", got)
	}
	if !strings.Contains(output.String(), "Connected to console of VM vm-1") {
		t.Fatalf("expected connection message, got %q", output.String())
	}
	if !strings.Contains(output.String(), "login: accepted\n") {
		t.Fatalf("expected streamed console output, got %q", output.String())
	}
}
