package websocket

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
)

func TestQGAVMCopyServiceUploadsFramesToGuestFile(t *testing.T) {
	guest := &recordingGuestFileClient{handle: 9}
	service := NewQGAVMCopyService(staticGuestFileClientResolver{client: guest})
	server := newVMCopyTestServer(t, service, VMCopyOptions{
		Direction: "upload",
		Path:      "/tmp/file",
		Mode:      "0644",
		Overwrite: true,
	})
	defer server.Close()

	conn, _, err := websocket.DefaultDialer.Dial("ws"+server.URL[len("http"):], nil)
	if err != nil {
		t.Fatalf("dial upload websocket: %v", err)
	}
	defer conn.Close()

	metadata, err := EncodeVMIOJSONFrame(VMIOFrameCopyMetadata, VMCopyMetadata{
		Path:   "/tmp/file",
		Size:   11,
		Mode:   "0644",
		SHA256: "ignored-by-unit-test",
	})
	if err != nil {
		t.Fatalf("encode metadata: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, metadata); err != nil {
		t.Fatalf("write metadata: %v", err)
	}

	_, ackFrame, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("read metadata ack: %v", err)
	}
	var ack VMIOAck
	frameType, err := DecodeVMIOJSONFrame(ackFrame, &ack)
	if err != nil {
		t.Fatalf("decode metadata ack: %v", err)
	}
	if frameType != VMIOFrameCopyAck || ack.Bytes != 0 {
		t.Fatalf("unexpected metadata ack type=%#x ack=%#v", frameType, ack)
	}

	if err := conn.WriteMessage(websocket.BinaryMessage, EncodeVMIODataFrame(VMIOFrameCopyData, []byte("hello "))); err != nil {
		t.Fatalf("write first data frame: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, EncodeVMIODataFrame(VMIOFrameCopyData, []byte("world"))); err != nil {
		t.Fatalf("write second data frame: %v", err)
	}
	eofFrame, err := EncodeVMIOJSONFrame(VMIOFrameCopyEOF, VMCopyEOF{Bytes: 11})
	if err != nil {
		t.Fatalf("encode eof: %v", err)
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, eofFrame); err != nil {
		t.Fatalf("write eof frame: %v", err)
	}

	_, finalAckFrame, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("read final ack: %v", err)
	}
	var finalAck VMIOAck
	frameType, err = DecodeVMIOJSONFrame(finalAckFrame, &finalAck)
	if err != nil {
		t.Fatalf("decode final ack: %v", err)
	}
	if frameType != VMIOFrameCopyAck || finalAck.Bytes != 11 {
		t.Fatalf("unexpected final ack type=%#x ack=%#v", frameType, finalAck)
	}

	if guest.openPath != "/tmp/file" || guest.openMode != "wb" {
		t.Fatalf("unexpected open call path=%s mode=%s", guest.openPath, guest.openMode)
	}
	if string(guest.written) != "hello world" {
		t.Fatalf("expected guest write payload %q, got %q", "hello world", string(guest.written))
	}
	if !guest.flushed || !guest.closed {
		t.Fatalf("expected guest file to be flushed and closed")
	}
}

func TestQGAVMCopyServiceDownloadsGuestFileFrames(t *testing.T) {
	guest := &recordingGuestFileClient{
		handle:     4,
		readChunks: [][]byte{[]byte("hello "), []byte("world")},
	}
	service := NewQGAVMCopyService(staticGuestFileClientResolver{client: guest})
	server := newVMCopyTestServer(t, service, VMCopyOptions{
		Direction: "download",
		Path:      "/tmp/file",
	})
	defer server.Close()

	conn, _, err := websocket.DefaultDialer.Dial("ws"+server.URL[len("http"):], nil)
	if err != nil {
		t.Fatalf("dial download websocket: %v", err)
	}
	defer conn.Close()

	_, metadataFrame, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("read metadata frame: %v", err)
	}
	var metadata VMCopyMetadata
	frameType, err := DecodeVMIOJSONFrame(metadataFrame, &metadata)
	if err != nil {
		t.Fatalf("decode metadata frame: %v", err)
	}
	if frameType != VMIOFrameCopyMetadata || metadata.Path != "/tmp/file" {
		t.Fatalf("unexpected metadata type=%#x metadata=%#v", frameType, metadata)
	}

	var downloaded []byte
	for {
		_, frame, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("read download frame: %v", err)
		}
		frameType, payload, err := DecodeVMIOFrame(frame)
		if err != nil {
			t.Fatalf("decode download frame: %v", err)
		}
		if frameType == VMIOFrameCopyData {
			downloaded = append(downloaded, payload...)
			continue
		}
		if frameType != VMIOFrameCopyEOF {
			t.Fatalf("unexpected download frame type %#x", frameType)
		}
		var eof VMCopyEOF
		if _, err := DecodeVMIOJSONFrame(frame, &eof); err != nil {
			t.Fatalf("decode eof frame: %v", err)
		}
		if eof.Bytes != 11 {
			t.Fatalf("expected eof byte count 11, got %d", eof.Bytes)
		}
		break
	}

	if string(downloaded) != "hello world" {
		t.Fatalf("expected downloaded payload %q, got %q", "hello world", string(downloaded))
	}
	if guest.openPath != "/tmp/file" || guest.openMode != "rb" || !guest.closed {
		t.Fatalf("unexpected guest file state path=%s mode=%s closed=%v", guest.openPath, guest.openMode, guest.closed)
	}
}

func newVMCopyTestServer(t *testing.T, service *QGAVMCopyService, options VMCopyOptions) *httptest.Server {
	t.Helper()
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade websocket: %v", err)
		}
		if err := service.HandleVMCopy(r.Context(), "vm-1", options, conn); err != nil {
			t.Errorf("handle vm copy: %v", err)
		}
	}))
}

type staticGuestFileClientResolver struct {
	client VMGuestFileClient
}

func (r staticGuestFileClientResolver) ResolveGuestFileClient(context.Context, string) (VMGuestFileClient, error) {
	return r.client, nil
}

type recordingGuestFileClient struct {
	mu         sync.Mutex
	handle     int
	openPath   string
	openMode   string
	written    []byte
	flushed    bool
	closed     bool
	readChunks [][]byte
	readIndex  int
}

func (c *recordingGuestFileClient) FileOpen(_ context.Context, path, mode string) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.openPath = path
	c.openMode = mode
	return c.handle, nil
}

func (c *recordingGuestFileClient) FileRead(_ context.Context, _ int, _ int) ([]byte, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.readIndex >= len(c.readChunks) {
		return nil, true, nil
	}
	chunk := append([]byte(nil), c.readChunks[c.readIndex]...)
	c.readIndex++
	return chunk, c.readIndex >= len(c.readChunks), nil
}

func (c *recordingGuestFileClient) FileWrite(_ context.Context, _ int, payload []byte) (int, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.written = append(c.written, payload...)
	return len(payload), false, nil
}

func (c *recordingGuestFileClient) FileFlush(context.Context, int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.flushed = true
	return nil
}

func (c *recordingGuestFileClient) FileClose(context.Context, int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	return nil
}
