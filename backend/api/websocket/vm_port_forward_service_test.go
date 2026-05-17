package websocket

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

type fakeGuestPortDialer struct {
	guestConn net.Conn
	dialedVM  string
	dialedTCP int
	dialErr   error
}

func (d *fakeGuestPortDialer) DialGuestPort(ctx context.Context, vmID string, port int) (net.Conn, error) {
	if d.dialErr != nil {
		return nil, d.dialErr
	}
	serviceConn, guestConn := net.Pipe()
	d.guestConn = guestConn
	d.dialedVM = vmID
	d.dialedTCP = port
	return serviceConn, nil
}

func TestVMPortForwardServiceMultiplexesWebSocketFramesToGuestConnection(t *testing.T) {
	dialer := &fakeGuestPortDialer{}
	service := NewVMPortForwardService(dialer)
	handler := NewWebSocketHandler(nil, nil, service, logrus.New())
	defer handler.Shutdown()

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, func(_ string, next http.HandlerFunc) http.Handler {
		return next
	})

	server := httptest.NewServer(router)
	defer server.Close()

	wsURL := "ws" + server.URL[len("http"):] + "/api/ws/vms/vm-1/port-forward?port=8080"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial websocket: %v", err)
	}
	defer conn.Close()

	openFrame, err := EncodeVMIOJSONFrame(VMIOFramePortForwardOpen, VMPortForwardOpen{
		ConnectionID: "conn-1",
		Port:         8080,
	})
	if err != nil {
		t.Fatalf("encode open frame: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, openFrame); err != nil {
		t.Fatalf("write open frame: %v", err)
	}

	var openAck VMPortForwardOpen
	frameType, err := readVMIOJSONTestFrame(conn, &openAck)
	if err != nil {
		t.Fatalf("read open ack: %v", err)
	}
	if frameType != VMIOFramePortForwardOpen {
		t.Fatalf("expected open ack frame, got %#x", frameType)
	}
	if openAck.ConnectionID != "conn-1" || openAck.Port != 8080 {
		t.Fatalf("unexpected open ack: %#v", openAck)
	}
	if dialer.dialedVM != "vm-1" || dialer.dialedTCP != 8080 {
		t.Fatalf("unexpected dial target vm=%q port=%d", dialer.dialedVM, dialer.dialedTCP)
	}

	dataFrame, err := EncodeVMPortForwardDataFrame("conn-1", []byte("request"))
	if err != nil {
		t.Fatalf("encode data frame: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, dataFrame); err != nil {
		t.Fatalf("write data frame: %v", err)
	}

	guestBuffer := make([]byte, len("request"))
	if err := dialer.guestConn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set guest read deadline: %v", err)
	}
	if _, err := dialer.guestConn.Read(guestBuffer); err != nil {
		t.Fatalf("read guest data: %v", err)
	}
	if string(guestBuffer) != "request" {
		t.Fatalf("expected guest request payload, got %q", guestBuffer)
	}

	if err := dialer.guestConn.SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set guest write deadline: %v", err)
	}
	if _, err := dialer.guestConn.Write([]byte("response")); err != nil {
		t.Fatalf("write guest response: %v", err)
	}

	messageType, frame, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("read websocket response: %v", err)
	}
	if messageType != websocket.BinaryMessage {
		t.Fatalf("expected binary response frame, got %d", messageType)
	}
	connectionID, payload, err := DecodeVMPortForwardDataFrame(frame)
	if err != nil {
		t.Fatalf("decode response frame: %v", err)
	}
	if connectionID != "conn-1" || string(payload) != "response" {
		t.Fatalf("unexpected response connection=%q payload=%q", connectionID, payload)
	}

	closeFrame, err := EncodeVMIOJSONFrame(VMIOFramePortForwardClose, VMPortForwardClose{
		ConnectionID: "conn-1",
		Reason:       "client_closed",
	})
	if err != nil {
		t.Fatalf("encode close frame: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, closeFrame); err != nil {
		t.Fatalf("write close frame: %v", err)
	}
}

func TestVMPortForwardServiceReportsGuestDialFailures(t *testing.T) {
	dialer := &fakeGuestPortDialer{dialErr: errors.New("guest network unavailable")}
	conn, cleanup := newVMPortForwardTestConnection(t, dialer)
	defer cleanup()

	openFrame, err := EncodeVMIOJSONFrame(VMIOFramePortForwardOpen, VMPortForwardOpen{
		ConnectionID: "conn-1",
		Port:         8080,
	})
	if err != nil {
		t.Fatalf("encode open frame: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, openFrame); err != nil {
		t.Fatalf("write open frame: %v", err)
	}

	var frameError VMIOError
	frameType, err := readVMIOJSONTestFrame(conn, &frameError)
	if err != nil {
		t.Fatalf("read error frame: %v", err)
	}
	if frameType != VMIOFramePortForwardError {
		t.Fatalf("expected port-forward error frame, got %#x", frameType)
	}
	if frameError.ConnectionID != "conn-1" || frameError.Code != "connect_failed" {
		t.Fatalf("unexpected error frame: %#v", frameError)
	}
}

func TestVMPortForwardServiceRejectsDataForUnknownConnections(t *testing.T) {
	dialer := &fakeGuestPortDialer{}
	conn, cleanup := newVMPortForwardTestConnection(t, dialer)
	defer cleanup()

	dataFrame, err := EncodeVMPortForwardDataFrame("missing", []byte("payload"))
	if err != nil {
		t.Fatalf("encode data frame: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, dataFrame); err != nil {
		t.Fatalf("write data frame: %v", err)
	}

	var frameError VMIOError
	frameType, err := readVMIOJSONTestFrame(conn, &frameError)
	if err != nil {
		t.Fatalf("read error frame: %v", err)
	}
	if frameType != VMIOFramePortForwardError {
		t.Fatalf("expected port-forward error frame, got %#x", frameType)
	}
	if frameError.ConnectionID != "missing" || frameError.Code != "unknown_connection" {
		t.Fatalf("unexpected error frame: %#v", frameError)
	}
}

func newVMPortForwardTestConnection(t *testing.T, dialer *fakeGuestPortDialer) (*websocket.Conn, func()) {
	t.Helper()

	service := NewVMPortForwardService(dialer)
	handler := NewWebSocketHandler(nil, nil, service, logrus.New())

	router := mux.NewRouter()
	handler.RegisterWebSocketRoutes(router, func(_ string, next http.HandlerFunc) http.Handler {
		return next
	})

	server := httptest.NewServer(router)
	wsURL := "ws" + server.URL[len("http"):] + "/api/ws/vms/vm-1/port-forward?port=8080"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		handler.Shutdown()
		server.Close()
		t.Fatalf("dial websocket: %v", err)
	}

	return conn, func() {
		_ = conn.Close()
		handler.Shutdown()
		server.Close()
	}
}

func readVMIOJSONTestFrame(conn *websocket.Conn, out interface{}) (byte, error) {
	if err := conn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		return 0, err
	}
	messageType, frame, err := conn.ReadMessage()
	if err != nil {
		return 0, err
	}
	if messageType != websocket.BinaryMessage {
		return 0, ErrVMIOInvalidFrame
	}
	return DecodeVMIOJSONFrame(frame, out)
}
