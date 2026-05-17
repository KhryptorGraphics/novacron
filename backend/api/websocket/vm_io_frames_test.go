package websocket

import (
	"bytes"
	"errors"
	"testing"
)

func TestEncodeDecodeVMIOJSONFrame(t *testing.T) {
	frame, err := EncodeVMIOJSONFrame(VMIOFrameCopyMetadata, VMCopyMetadata{
		Path:   "/tmp/file",
		Size:   123,
		Mode:   "0644",
		SHA256: "abc123",
	})
	if err != nil {
		t.Fatalf("encode metadata frame: %v", err)
	}

	var decoded VMCopyMetadata
	frameType, err := DecodeVMIOJSONFrame(frame, &decoded)
	if err != nil {
		t.Fatalf("decode metadata frame: %v", err)
	}

	if frameType != VMIOFrameCopyMetadata {
		t.Fatalf("expected frame type %#x, got %#x", VMIOFrameCopyMetadata, frameType)
	}
	if decoded.Path != "/tmp/file" || decoded.Size != 123 || decoded.Mode != "0644" || decoded.SHA256 != "abc123" {
		t.Fatalf("unexpected decoded metadata: %#v", decoded)
	}
}

func TestEncodeDecodeVMIODataFrame(t *testing.T) {
	payload := []byte{0x00, 0x01, 0x02, 0xff}
	frame := EncodeVMIODataFrame(VMIOFrameCopyData, payload)
	frameType, decoded, err := DecodeVMIOFrame(frame)
	if err != nil {
		t.Fatalf("decode data frame: %v", err)
	}

	if frameType != VMIOFrameCopyData {
		t.Fatalf("expected frame type %#x, got %#x", VMIOFrameCopyData, frameType)
	}
	if !bytes.Equal(decoded, payload) {
		t.Fatalf("expected payload %v, got %v", payload, decoded)
	}

	payload[0] = 0x99
	if decoded[0] == 0x99 {
		t.Fatalf("decoded payload should not alias source payload")
	}
}

func TestDecodeVMIOFrameRejectsEmptyMessage(t *testing.T) {
	_, _, err := DecodeVMIOFrame(nil)
	if !errors.Is(err, ErrVMIOEmptyFrame) {
		t.Fatalf("expected ErrVMIOEmptyFrame, got %v", err)
	}
}

func TestEncodeDecodeVMPortForwardDataFrame(t *testing.T) {
	payload := []byte("GET / HTTP/1.1\r\n\r\n")
	frame, err := EncodeVMPortForwardDataFrame("conn-1", payload)
	if err != nil {
		t.Fatalf("encode port-forward data frame: %v", err)
	}

	connectionID, decoded, err := DecodeVMPortForwardDataFrame(frame)
	if err != nil {
		t.Fatalf("decode port-forward data frame: %v", err)
	}

	if connectionID != "conn-1" {
		t.Fatalf("expected connection id conn-1, got %s", connectionID)
	}
	if !bytes.Equal(decoded, payload) {
		t.Fatalf("expected payload %q, got %q", payload, decoded)
	}
}

func TestEncodeVMPortForwardDataFrameRejectsInvalidConnectionID(t *testing.T) {
	for _, connectionID := range []string{"", string(bytes.Repeat([]byte("a"), 256))} {
		if _, err := EncodeVMPortForwardDataFrame(connectionID, []byte("payload")); err == nil {
			t.Fatalf("expected error for connection id length %d", len(connectionID))
		}
	}
}

func TestDecodeVMPortForwardDataFrameRejectsMalformedFrame(t *testing.T) {
	for _, frame := range [][]byte{
		{},
		{VMIOFramePortForwardData},
		{VMIOFramePortForwardOpen, 1, 'a'},
		{VMIOFramePortForwardData, 3, 'a'},
	} {
		if _, _, err := DecodeVMPortForwardDataFrame(frame); err == nil {
			t.Fatalf("expected error for malformed frame %v", frame)
		}
	}
}
