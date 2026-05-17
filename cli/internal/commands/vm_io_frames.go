package commands

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	cliVMIOFrameCopyMetadata byte = 0x01
	cliVMIOFrameCopyData     byte = 0x02
	cliVMIOFrameCopyEOF      byte = 0x03
	cliVMIOFrameCopyError    byte = 0x04
	cliVMIOFrameCopyAck      byte = 0x05

	cliVMIOFramePortForwardOpen      byte = 0x10
	cliVMIOFramePortForwardData      byte = 0x11
	cliVMIOFramePortForwardClose     byte = 0x12
	cliVMIOFramePortForwardError     byte = 0x13
	cliVMIOFramePortForwardHeartbeat byte = 0x14
)

var errCLIVMIOInvalidFrame = errors.New("vm io frame is invalid")

type cliVMCopyMetadata struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	Mode   string `json:"mode,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
}

type cliVMCopyEOF struct {
	SHA256 string `json:"sha256,omitempty"`
	Bytes  int64  `json:"bytes"`
}

type cliVMIOAck struct {
	Bytes int64 `json:"bytes"`
}

type cliVMPortForwardOpen struct {
	ConnectionID string `json:"connectionId"`
	Port         int    `json:"port"`
}

type cliVMPortForwardClose struct {
	ConnectionID string `json:"connectionId"`
	Reason       string `json:"reason,omitempty"`
}

type cliVMIOError struct {
	ConnectionID string `json:"connectionId,omitempty"`
	Code         string `json:"code"`
	Message      string `json:"message"`
}

func encodeCLIVMIOJSONFrame(frameType byte, value interface{}) ([]byte, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("marshal vm io frame: %w", err)
	}
	return encodeCLIVMIODataFrame(frameType, payload), nil
}

func encodeCLIVMIODataFrame(frameType byte, payload []byte) []byte {
	frame := make([]byte, len(payload)+1)
	frame[0] = frameType
	copy(frame[1:], payload)
	return frame
}

func decodeCLIVMIOFrame(data []byte) (byte, []byte, error) {
	if len(data) == 0 {
		return 0, nil, errCLIVMIOInvalidFrame
	}
	payload := make([]byte, len(data)-1)
	copy(payload, data[1:])
	return data[0], payload, nil
}

func decodeCLIVMIOJSONFrame(data []byte, out interface{}) (byte, error) {
	frameType, payload, err := decodeCLIVMIOFrame(data)
	if err != nil {
		return 0, err
	}
	if err := json.Unmarshal(payload, out); err != nil {
		return 0, fmt.Errorf("unmarshal vm io frame: %w", err)
	}
	return frameType, nil
}

func encodeCLIVMPortForwardDataFrame(connectionID string, payload []byte) ([]byte, error) {
	if len(connectionID) == 0 || len(connectionID) > 255 {
		return nil, errCLIVMIOInvalidFrame
	}
	frame := make([]byte, len(payload)+2+len(connectionID))
	frame[0] = cliVMIOFramePortForwardData
	frame[1] = byte(len(connectionID))
	copy(frame[2:], connectionID)
	copy(frame[2+len(connectionID):], payload)
	return frame, nil
}

func decodeCLIVMPortForwardDataFrame(data []byte) (string, []byte, error) {
	frameType, payload, err := decodeCLIVMIOFrame(data)
	if err != nil {
		return "", nil, err
	}
	if frameType != cliVMIOFramePortForwardData || len(payload) == 0 {
		return "", nil, errCLIVMIOInvalidFrame
	}
	connectionIDLength := int(payload[0])
	if connectionIDLength == 0 || len(payload) < connectionIDLength+1 {
		return "", nil, errCLIVMIOInvalidFrame
	}
	connectionID := string(payload[1 : connectionIDLength+1])
	decodedPayload := make([]byte, len(payload)-connectionIDLength-1)
	copy(decodedPayload, payload[connectionIDLength+1:])
	return connectionID, decodedPayload, nil
}
