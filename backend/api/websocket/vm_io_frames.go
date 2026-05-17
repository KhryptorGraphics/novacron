package websocket

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	VMIOFrameCopyMetadata byte = 0x01
	VMIOFrameCopyData     byte = 0x02
	VMIOFrameCopyEOF      byte = 0x03
	VMIOFrameCopyError    byte = 0x04
	VMIOFrameCopyAck      byte = 0x05

	VMIOFramePortForwardOpen      byte = 0x10
	VMIOFramePortForwardData      byte = 0x11
	VMIOFramePortForwardClose     byte = 0x12
	VMIOFramePortForwardError     byte = 0x13
	VMIOFramePortForwardHeartbeat byte = 0x14
)

var (
	ErrVMIOEmptyFrame     = errors.New("vm io frame is empty")
	ErrVMIOInvalidFrame   = errors.New("vm io frame is invalid")
	ErrVMIOInvalidFrameID = errors.New("vm io connection id is invalid")
)

type VMCopyMetadata struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	Mode   string `json:"mode,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
}

type VMCopyEOF struct {
	SHA256 string `json:"sha256,omitempty"`
	Bytes  int64  `json:"bytes"`
}

type VMIOError struct {
	ConnectionID string `json:"connectionId,omitempty"`
	Code         string `json:"code"`
	Message      string `json:"message"`
}

type VMIOAck struct {
	Bytes int64 `json:"bytes"`
}

type VMPortForwardOpen struct {
	ConnectionID string `json:"connectionId"`
	Port         int    `json:"port"`
}

type VMPortForwardClose struct {
	ConnectionID string `json:"connectionId"`
	Reason       string `json:"reason,omitempty"`
}

type VMPortForwardHeartbeat struct {
	Timestamp string `json:"timestamp"`
}

func EncodeVMIOJSONFrame(frameType byte, value interface{}) ([]byte, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("marshal vm io frame: %w", err)
	}
	return EncodeVMIODataFrame(frameType, payload), nil
}

func EncodeVMIODataFrame(frameType byte, payload []byte) []byte {
	frame := make([]byte, len(payload)+1)
	frame[0] = frameType
	copy(frame[1:], payload)
	return frame
}

func DecodeVMIOFrame(data []byte) (byte, []byte, error) {
	if len(data) == 0 {
		return 0, nil, ErrVMIOEmptyFrame
	}

	payload := make([]byte, len(data)-1)
	copy(payload, data[1:])
	return data[0], payload, nil
}

func DecodeVMIOJSONFrame(data []byte, out interface{}) (byte, error) {
	frameType, payload, err := DecodeVMIOFrame(data)
	if err != nil {
		return 0, err
	}
	if err := json.Unmarshal(payload, out); err != nil {
		return 0, fmt.Errorf("unmarshal vm io frame: %w", err)
	}
	return frameType, nil
}

func EncodeVMPortForwardDataFrame(connectionID string, payload []byte) ([]byte, error) {
	if len(connectionID) == 0 || len(connectionID) > 255 {
		return nil, ErrVMIOInvalidFrameID
	}

	frame := make([]byte, len(payload)+2+len(connectionID))
	frame[0] = VMIOFramePortForwardData
	frame[1] = byte(len(connectionID))
	copy(frame[2:], connectionID)
	copy(frame[2+len(connectionID):], payload)
	return frame, nil
}

func DecodeVMPortForwardDataFrame(data []byte) (string, []byte, error) {
	frameType, payload, err := DecodeVMIOFrame(data)
	if err != nil {
		return "", nil, err
	}
	if frameType != VMIOFramePortForwardData {
		return "", nil, fmt.Errorf("%w: unexpected port-forward frame type %#x", ErrVMIOInvalidFrame, frameType)
	}
	if len(payload) == 0 {
		return "", nil, fmt.Errorf("%w: missing connection id length", ErrVMIOInvalidFrame)
	}

	connectionIDLength := int(payload[0])
	if connectionIDLength == 0 || len(payload) < connectionIDLength+1 {
		return "", nil, ErrVMIOInvalidFrameID
	}

	connectionID := string(payload[1 : connectionIDLength+1])
	decodedPayload := make([]byte, len(payload)-connectionIDLength-1)
	copy(decodedPayload, payload[connectionIDLength+1:])
	return connectionID, decodedPayload, nil
}
