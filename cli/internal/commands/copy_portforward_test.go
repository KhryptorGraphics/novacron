package commands

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestCopyCommandReportsMissingBackendContract(t *testing.T) {
	cmd := NewCopyCommand()
	cmd.SetArgs([]string{"local.txt", "vm-1:/tmp/local.txt"})

	err := cmd.Execute()
	if err == nil {
		t.Fatalf("expected copy contract error")
	}
	for _, expected := range []string{
		"backend contract is not implemented",
		"docs/api/vm-io-contracts.md",
		"novacron-lmh",
	} {
		if !strings.Contains(err.Error(), expected) {
			t.Fatalf("expected error to contain %q, got %v", expected, err)
		}
	}
}

func TestPortForwardCommandForwardsLocalTCPOverWebSocket(t *testing.T) {
	withTempHome(t)

	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ws/vms/vm-1/port-forward" {
			t.Fatalf("expected port-forward websocket path, got %s", r.URL.Path)
		}
		if r.URL.Query().Get("port") != "80" {
			t.Fatalf("expected remote port 80, got query %s", r.URL.RawQuery)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		_, frame, err := conn.ReadMessage()
		if err != nil {
			t.Errorf("read open frame: %v", err)
			return
		}
		frameType, payload, err := decodeCLIVMIOFrame(frame)
		if err != nil {
			t.Errorf("decode open frame: %v", err)
			return
		}
		if frameType != cliVMIOFramePortForwardOpen || !bytes.Contains(payload, []byte(`"connectionId"`)) {
			t.Errorf("unexpected open frame type=%#x payload=%s", frameType, payload)
			return
		}
		connectionID := extractConnectionID(t, payload)
		openAck, err := encodeCLIVMIOJSONFrame(cliVMIOFramePortForwardOpen, cliVMPortForwardOpen{
			ConnectionID: connectionID,
			Port:         80,
		})
		if err != nil {
			t.Errorf("encode open ack: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, openAck); err != nil {
			t.Errorf("write open ack: %v", err)
			return
		}

		_, dataFrame, err := conn.ReadMessage()
		if err != nil {
			t.Errorf("read data frame: %v", err)
			return
		}
		gotConnectionID, requestPayload, err := decodeCLIVMPortForwardDataFrame(dataFrame)
		if err != nil {
			t.Errorf("decode data frame: %v", err)
			return
		}
		if gotConnectionID != connectionID || string(requestPayload) != "ping" {
			t.Errorf("unexpected data connection=%q payload=%q", gotConnectionID, requestPayload)
			return
		}

		responseFrame, err := encodeCLIVMPortForwardDataFrame(connectionID, []byte("pong"))
		if err != nil {
			t.Errorf("encode response frame: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, responseFrame); err != nil {
			t.Errorf("write response frame: %v", err)
			return
		}
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var output bytes.Buffer
	cmd := NewPortForwardCommand()
	cmd.SetContext(ctx)
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"vm-1", "0:80"})

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Execute()
	}()

	localPort := waitForForwardedLocalPort(t, &output)
	localConn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", localPort), time.Second)
	if err != nil {
		t.Fatalf("dial local forwarded port: %v", err)
	}
	defer localConn.Close()

	if _, err := localConn.Write([]byte("ping")); err != nil {
		t.Fatalf("write local request: %v", err)
	}
	buffer := make([]byte, len("pong"))
	if err := localConn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set local read deadline: %v", err)
	}
	if _, err := localConn.Read(buffer); err != nil {
		t.Fatalf("read local response: %v", err)
	}
	if string(buffer) != "pong" {
		t.Fatalf("expected forwarded response pong, got %q", buffer)
	}

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("port-forward command failed: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("port-forward command did not exit after context cancellation")
	}
}

func extractConnectionID(t *testing.T, payload []byte) string {
	t.Helper()
	matches := regexp.MustCompile(`"connectionId":"([^"]+)"`).FindSubmatch(payload)
	if len(matches) != 2 {
		t.Fatalf("connectionId missing from payload %s", payload)
	}
	return string(matches[1])
}

func waitForForwardedLocalPort(t *testing.T, output *bytes.Buffer) string {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	pattern := regexp.MustCompile(`127\.0\.0\.1:(\d+)`)
	for time.Now().Before(deadline) {
		if matches := pattern.FindStringSubmatch(output.String()); len(matches) == 2 {
			return matches[1]
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("forwarded local port not reported in output:\n%s", output.String())
	return ""
}
