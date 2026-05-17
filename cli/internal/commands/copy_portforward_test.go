package commands

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestCopyCommandUploadsLocalFileOverWebSocket(t *testing.T) {
	withTempHome(t)

	sourcePath := filepath.Join(t.TempDir(), "local.txt")
	sourceContent := []byte("hello world")
	if err := os.WriteFile(sourcePath, sourceContent, 0o644); err != nil {
		t.Fatalf("write source file: %v", err)
	}

	uploaded := make(chan []byte, 1)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ws/vms/vm-1/copy" {
			t.Fatalf("expected copy websocket path, got %s", r.URL.Path)
		}
		if r.URL.Query().Get("direction") != "upload" || r.URL.Query().Get("path") != "/tmp/local.txt" {
			t.Fatalf("unexpected copy query %s", r.URL.RawQuery)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		_, frame, err := conn.ReadMessage()
		if err != nil {
			t.Errorf("read metadata frame: %v", err)
			return
		}
		var metadata cliVMCopyMetadata
		frameType, err := decodeCLIVMIOJSONFrame(frame, &metadata)
		if err != nil {
			t.Errorf("decode metadata frame: %v", err)
			return
		}
		if frameType != cliVMIOFrameCopyMetadata || metadata.Path != "/tmp/local.txt" || metadata.Size != int64(len(sourceContent)) {
			t.Errorf("unexpected metadata frame type=%#x metadata=%+v", frameType, metadata)
			return
		}
		if err := writeCopyAck(conn, 0); err != nil {
			t.Errorf("write metadata ack: %v", err)
			return
		}

		var got bytes.Buffer
		for {
			_, frame, err := conn.ReadMessage()
			if err != nil {
				t.Errorf("read upload frame: %v", err)
				return
			}
			frameType, payload, err := decodeCLIVMIOFrame(frame)
			if err != nil {
				t.Errorf("decode upload frame: %v", err)
				return
			}
			switch frameType {
			case cliVMIOFrameCopyData:
				got.Write(payload)
			case cliVMIOFrameCopyEOF:
				var eof cliVMCopyEOF
				if _, err := decodeCLIVMIOJSONFrame(frame, &eof); err != nil {
					t.Errorf("decode eof frame: %v", err)
					return
				}
				if eof.Bytes != int64(len(sourceContent)) {
					t.Errorf("unexpected eof byte count %d", eof.Bytes)
					return
				}
				if err := writeCopyAck(conn, int64(got.Len())); err != nil {
					t.Errorf("write final ack: %v", err)
					return
				}
				uploaded <- got.Bytes()
				return
			default:
				t.Errorf("unexpected upload frame type %#x", frameType)
				return
			}
		}
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	cmd := NewCopyCommand()
	cmd.SetArgs([]string{sourcePath, "vm-1:/tmp/local.txt"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("copy upload command failed: %v", err)
	}

	select {
	case got := <-uploaded:
		if !bytes.Equal(got, sourceContent) {
			t.Fatalf("expected uploaded content %q, got %q", sourceContent, got)
		}
	case <-time.After(time.Second):
		t.Fatal("upload was not received by websocket server")
	}
}

func TestCopyCommandDownloadsRemoteFileOverWebSocket(t *testing.T) {
	withTempHome(t)

	remoteContent := []byte("remote data\n")
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ws/vms/vm-1/copy" {
			t.Fatalf("expected copy websocket path, got %s", r.URL.Path)
		}
		if r.URL.Query().Get("direction") != "download" || r.URL.Query().Get("path") != "/var/log/app.log" {
			t.Fatalf("unexpected copy query %s", r.URL.RawQuery)
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		defer conn.Close()

		metadataFrame, err := encodeCLIVMIOJSONFrame(cliVMIOFrameCopyMetadata, cliVMCopyMetadata{
			Path: "/var/log/app.log",
			Size: int64(len(remoteContent)),
			Mode: "0644",
		})
		if err != nil {
			t.Errorf("encode metadata: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, metadataFrame); err != nil {
			t.Errorf("write metadata: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, encodeCLIVMIODataFrame(cliVMIOFrameCopyData, remoteContent)); err != nil {
			t.Errorf("write data: %v", err)
			return
		}
		eofFrame, err := encodeCLIVMIOJSONFrame(cliVMIOFrameCopyEOF, cliVMCopyEOF{Bytes: int64(len(remoteContent))})
		if err != nil {
			t.Errorf("encode eof: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.BinaryMessage, eofFrame); err != nil {
			t.Errorf("write eof: %v", err)
			return
		}
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	destinationPath := filepath.Join(t.TempDir(), "app.log")
	cmd := NewCopyCommand()
	cmd.SetArgs([]string{"vm-1:/var/log/app.log", destinationPath})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("copy download command failed: %v", err)
	}

	got, err := os.ReadFile(destinationPath)
	if err != nil {
		t.Fatalf("read downloaded file: %v", err)
	}
	if !bytes.Equal(got, remoteContent) {
		t.Fatalf("expected downloaded content %q, got %q", remoteContent, got)
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

func writeCopyAck(conn *websocket.Conn, bytes int64) error {
	frame, err := encodeCLIVMIOJSONFrame(cliVMIOFrameCopyAck, cliVMIOAck{Bytes: bytes})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.BinaryMessage, frame)
}
